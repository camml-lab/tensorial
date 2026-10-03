import abc
from collections.abc import Sequence
import math

from typing import TYPE_CHECKING, Final, Generic, NamedTuple, TypedDict, TypeVar

import jraph
import reax
from typing_extensions import override

from ... import data
from . import _batching, _common, _dataloader

if TYPE_CHECKING:
    import jax

    import tensorial.data
    from tensorial import gcnn

__all__ = (
    "GraphDataset",
    "GraphDataModule",
    "Split",
    "SplitStrategy",
    "RandomSplit",
    "KFoldSplit",
    "PreSplit",
    "PreSplitDataset",
)

# ``D`` is the *input* the strategy consumes: a single dataset for
# ``RandomSplit`` / ``KFoldSplit``, a ``PreSplitDataset`` mapping for
# ``PreSplit``.
D = TypeVar("D")


GraphDataset = Sequence[jraph.GraphsTuple]
GraphDataFetcher = data.DataFetcher[D]


class Split(NamedTuple):
    """The result of a split: train/val/test datasets, any of which may be absent."""

    train: GraphDataset | None = None
    val: GraphDataset | None = None
    test: GraphDataset | None = None


class PreSplitDataset(TypedDict, total=False):
    """Datasets the user has already split.

    Keys are optional and values may be ``None``; at least one must be
    non-``None`` (enforced by ``GraphDataModule.setup``).
    """

    train: GraphDataset | None
    val: GraphDataset | None
    test: GraphDataset | None


class SplitStrategy(Generic[D], abc.ABC):
    """Abstract base for dataset splitting strategies.

    A ``SplitStrategy`` is a **pure function** from a source dataset to a
    :class:`Split`.  It holds no reference to the dataset it operates on --
    that is passed to :meth:`split` -- so the same strategy instance can be
    reused and tested independently of any particular data.

    The strategy is consumed by ``GraphDataModule``, which holds the source
    dataset and is responsible for everything downstream: padding, batching,
    and loader construction.

    Contract:

    * ``split`` must be **deterministic** given the same ``dataset``, ``rngs``,
      and ``stage``.  Concretely, every process in a distributed run must
      observe the same split, otherwise different ranks will train on
      different data.
    * ``split`` must be **pure**: calling it twice with the same arguments
      must return equal results.  Caching the result is the caller's
      responsibility, not the strategy's.
    """

    @abc.abstractmethod
    def split(self, dataset: D, *, rngs, stage: "reax.Stage") -> Split:
        """Produce the :class:`Split` of datasets.

        Args:
            dataset: The source data to split.  Its type depends on the
                strategy (a single dataset, a ``PreSplitDataset`` mapping,
                etc.); the type parameter ``D`` makes that explicit.
            rngs: Random number generators for any randomised splitting.
                Passing these in (rather than storing them on the strategy)
                keeps the strategy free of hidden randomness and makes the
                split reproducible from the caller's seed.
            stage: The reax stage.  Strategies that behave differently under
                ``fit`` vs. ``test`` vs. ``predict`` can branch on this.

        Returns:
            A ``Split`` of datasets in train, val, test order.  Any of the
            three may be ``None``.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class RandomSplit(SplitStrategy[GraphDataset]):
    """Split a single dataset into train/val/test by random partition."""

    def __init__(self, fractions: Sequence[float] = (0.85, 0.05, 0.1)) -> None:
        if len(fractions) != 3:
            raise ValueError(f"Expected 3 split fractions (train, val, test), got {len(fractions)}")
        if not math.isclose(sum(fractions), 1.0):
            raise ValueError(f"Split fractions must sum to 1.0, got {sum(fractions)}")
        if any(f < 0 for f in fractions):
            raise ValueError(f"Split fractions must be non-negative, got {fractions}")

        self._fractions: Final[tuple[float, ...]] = tuple(fractions)

    @override
    def split(self, dataset: GraphDataset, *, rngs, stage) -> Split:
        return Split(*reax.data.random_split(rngs, dataset, lengths=self._fractions))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(fractions={self._fractions})"


class KFoldSplit(SplitStrategy[GraphDataset]):
    """Hold out one fold as the validation set for k-fold cross-validation.

    The test set is carved out first using ``test_fraction``.  The remaining
    data is partitioned into ``n_folds`` equal folds; fold ``fold`` becomes
    the validation set, and the rest becomes the training set.  The train
    and validation *fractions* are therefore not configurable in this mode
    -- the train/val split is controlled exclusively by ``n_folds`` and
    ``fold``.
    """

    def __init__(
        self, fold: int, n_folds: int = 5, *, test_fraction: float = 0.1, seed: int | None = 42
    ) -> None:
        if not 0 <= fold < n_folds:
            raise ValueError(f"fold must be in [0, {n_folds}), got {fold}")
        if not 0.0 < test_fraction < 1.0:
            raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")

        # Params
        self._fold: Final[int] = fold
        self._n_folds: Final[int] = n_folds
        self._test_fraction: Final[float] = test_fraction
        self._seed: Final[int | None] = seed

    def split(self, dataset: GraphDataset, *, rngs, stage) -> Split:
        rest, test = reax.data.random_split(
            rngs,
            dataset,
            lengths=(1.0 - self._test_fraction, self._test_fraction),
        )
        kfold = reax.data.KFold(n_splits=self._n_folds, shuffle=True, seed=self._seed)
        train, val = kfold.get_fold(rest, self._fold)
        return Split(train=train, val=val, test=test)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(fold={self._fold}, "
            f"n_folds={self._n_folds}, seed={self._seed}, "
            f"test_fraction={self._test_fraction})"
        )


class PreSplit(SplitStrategy[PreSplitDataset]):
    """Use datasets that the user has already split.

    The "input" is a :class:`PreSplitDataset` (a ``TypedDict`` with optional
    ``train`` / ``val`` / ``test`` keys, each value ``None`` or a dataset).
    ``split`` just unpacks the mapping.  At least one of the three must be
    non-``None``; this is checked by ``GraphDataModule.setup``.
    """

    def split(self, dataset: PreSplitDataset, *, rngs, stage) -> Split:
        return Split(
            train=dataset.get("train"),
            val=dataset.get("val"),
            test=dataset.get("test"),
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class GraphDataModule(Generic[D], reax.DataModule[jraph.GraphsTuple, jraph.GraphsTuple]):
    """A data module that serves ``jraph.GraphsTuple`` datasets.

    Construct via :meth:`from_random_split`, :meth:`from_kfold`, or
    :meth:`from_datasets` rather than the (strategy-based) ``__init__``.
    """

    def __init__(
        self,
        dataset: "tensorial.data.DataFetcher[D] | D",
        strategy: SplitStrategy[D],
        *,
        batch_size: int = 32,
        batch_mode: "gcnn.data.BatchMode | str" = _common.BatchMode.IMPLICIT,
        pad_to_multiple: "int | str | jax.Device | None" = None,
    ) -> None:
        super().__init__()
        self._fetcher, self._dataset = self._init_fetcher_dataset(dataset)

        # Params
        self._strategy: Final[SplitStrategy[D]] = strategy
        self._batch_size: Final[int] = batch_size
        self._batch_mode = _common.BatchMode(batch_mode)
        self._pad_to_multiple: Final = pad_to_multiple

        # State
        self.data_train: GraphDataset | None = None
        self.data_val: GraphDataset | None = None
        self.data_test: GraphDataset | None = None
        self._max_padding: "gcnn.data.GraphPadding | None" = None
        self._setup_done = False

    @staticmethod
    def _init_fetcher_dataset(data_fetcher):
        if isinstance(data_fetcher, data.DataFetcher):
            return data_fetcher, None

        return None, data_fetcher

    @property
    def batch_size(self) -> int:
        """The batch size, expressed as a per-device figure (nothing multiplies or
        divides it).  Read-only: nothing mutates the batch size after the loaders
        are constructed, so there is no per-device override knob exposed here.
        """
        return self._batch_size

    @override
    def prepare_data(self) -> None:
        """Fetch the dataset via the configured data fetcher and store it in the module."""
        if self._fetcher is not None:
            self._dataset = self._fetcher.fetch()
        else:
            assert self._dataset is not None, "No dataset or fetcher provided"

    @override
    def setup(self, stage: reax.Stage, /) -> None:
        assert self._dataset is not None, "prepare_data() must be called before setup()"

        if self._setup_done:
            return
        split = self._strategy.split(self._dataset, rngs=self.rngs, stage=stage)
        self.data_train, self.data_val, self.data_test = split
        if self.data_train is None and self.data_val is None and self.data_test is None:
            raise reax.exceptions.MisconfigurationException(
                f"SplitStrategy {self._strategy!r} returned all None datasets, have no data"
            )
        self._max_padding = self._compute_shared_padding()
        self._setup_done = True

    @override
    def train_dataloader(self) -> reax.DataLoader:
        """Create and return the train dataloader.

        Returns:
            The train dataloader.
        """
        if not self._setup_done:
            raise reax.exceptions.MisconfigurationException(
                "Must call setup() before requesting the train dataloader"
            )
        if self.data_train is None:
            raise reax.exceptions.MisconfigurationException(
                f"Strategy {self._strategy!r} produced no training data; "
                "cannot construct a train dataloader"
            )

        return _dataloader.GraphLoader(
            self.data_train,
            batch_size=self.batch_size,
            padding=self._max_padding,
            pad=True,
            batch_mode=self._batch_mode,
        )

    @override
    def val_dataloader(self) -> reax.DataLoader:
        """Create and return the validation dataloader.

        Returns:
            The validation dataloader.
        """
        if not self._setup_done:
            raise reax.exceptions.MisconfigurationException(
                "Must call setup() before requesting the validation dataloader"
            )
        if self.data_val is None:
            raise reax.exceptions.MisconfigurationException(
                f"Strategy {self._strategy!r} produced no validation data; "
                "cannot construct a validation dataloader"
            )

        return _dataloader.GraphLoader(
            self.data_val,
            batch_size=self.batch_size,
            shuffle=False,
            padding=self._max_padding,
            pad=True,
            batch_mode=self._batch_mode,
        )

    @override
    def test_dataloader(self) -> reax.DataLoader:
        """Create and return the test dataloader.

        Returns:
            The test dataloader.
        """
        if not self._setup_done:
            raise reax.exceptions.MisconfigurationException(
                "Must call setup() before requesting the test dataloader"
            )
        if self.data_test is None:
            raise reax.exceptions.MisconfigurationException(
                f"Strategy {self._strategy!r} produced no test data; "
                "cannot construct a test dataloader"
            )

        return _dataloader.GraphLoader(
            self.data_test,
            batch_size=self.batch_size,
            shuffle=False,
            padding=self._max_padding,
            pad=True,
            batch_mode=self._batch_mode,
        )

    def _compute_shared_padding(self) -> "gcnn.data.GraphPadding":
        # Called only from ``setup()`` after the "all-None" guard has raised,
        # so at least one split is non-``None`` and ``paddings`` is guaranteed
        # non-empty.
        paddings = []
        for graphs in (self.data_train, self.data_val, self.data_test):
            if graphs is not None:
                if self._batch_mode is _common.BatchMode.IMPLICIT:
                    paddings.append(
                        _batching.GraphBatcher.calculate_padding(
                            graphs,
                            self._batch_size,
                            pad_to_multiple=self._pad_to_multiple,
                        )
                    )
                else:
                    paddings.append(_batching.GraphBatcher.calculate_padding(graphs, 1))

        return _batching.max_padding(*paddings)

    @classmethod
    def from_random_split(
        cls,
        dataset: "tensorial.data.DataFetcher[D] | GraphDataset",
        train_val_test_split: Sequence[float] = (0.85, 0.05, 0.1),
        *,
        batch_size: int = 32,
        batch_mode: "gcnn.data.BatchMode | str" = _common.BatchMode.IMPLICIT,
        pad_to_multiple: "int | str | jax.Device | None" = None,
    ) -> "GraphDataModule":
        """Build a module that splits one dataset into train/val/test at random.

        Args:
            dataset: The source dataset to split.  Either a plain dataset
                (wrapped in a :class:`PassthroughFetcher`) or an existing
                ``data_fetcher.DataFetcher``.
        """
        return cls(
            dataset,
            RandomSplit(fractions=train_val_test_split),
            batch_size=batch_size,
            batch_mode=batch_mode,
            pad_to_multiple=pad_to_multiple,
        )

    @classmethod
    def from_kfold(
        cls,
        dataset: "tensorial.data.DataFetcher[D] | GraphDataset",
        fold: int,
        n_folds: int = 5,
        *,
        seed: int | None = 42,
        test_fraction: float = 0.1,
        batch_size: int = 32,
        batch_mode: "gcnn.data.BatchMode | str" = _common.BatchMode.IMPLICIT,
        pad_to_multiple: "int | str | jax.Device | None" = None,
    ) -> "GraphDataModule":
        """Build a module holding out one fold for k-fold cross-validation.

        Only ``train_val_test_split[2]`` (the test fraction) is used; the
        train/val split is driven by ``n_folds`` and ``fold``, so
        ``train_val_test_split[0]`` and ``[1]`` are ignored.

        Args:
            dataset: The source dataset to split.  Either a plain dataset
                (wrapped in a :class:`PassthroughFetcher`) or an existing
                ``data_fetcher.DataFetcher``.
        """
        return cls(
            dataset,
            KFoldSplit(fold=fold, n_folds=n_folds, seed=seed, test_fraction=test_fraction),
            batch_size=batch_size,
            batch_mode=batch_mode,
            pad_to_multiple=pad_to_multiple,
        )

    @classmethod
    def from_datasets(
        # pylint: disable=arguments-differ
        cls,
        dataset: "tensorial.data.DataFetcher[gcnn.data.PreSplitDataset] | PreSplitDataset",
        *,
        batch_size: int = 32,
        batch_mode: "gcnn.data.BatchMode | str" = _common.BatchMode.IMPLICIT,
        pad_to_multiple: "int | str | jax.Device | None" = None,
    ) -> "GraphDataModule":
        """Build a module from datasets the user has already split."""
        return cls(
            dataset,
            PreSplit(),
            batch_size=batch_size,
            batch_mode=batch_mode,
            pad_to_multiple=pad_to_multiple,
        )
