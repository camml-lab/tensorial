import abc
from collections.abc import Callable, Sequence
import math
from typing import TYPE_CHECKING, Any, Final, Generic, NamedTuple, TypedDict, TypeVar, cast

import jraph
import numpy as np
import reax
from typing_extensions import override

from ... import data
from .. import keys as gcnn_keys
from ..atomic import keys as atomic_keys
from . import _batching, _common, _dataloader

if TYPE_CHECKING:
    import jax

    from tensorial import gcnn
    import tensorial.data

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


# Keys that a single graph may legitimately omit but which are required for a
# mixed molecule + crystal dataset to be batch-able, together with a
# per-graph-size factory for the value to fill in.  This is the set of keys
# whose presence is a *convention* (not an invariant): an open-boundary molecule
# graph is allowed to lack them, but every graph in a batch must agree.
_OPTIONAL_EDGE_KEYS: Final[dict[str, Callable[[int], np.ndarray]]] = {
    gcnn_keys.EDGE_CELL_SHIFTS: lambda n: np.zeros((n, 3), dtype=np.float32),
}
_OPTIONAL_GLOBAL_KEYS: Final[dict[str, Callable[[int], np.ndarray]]] = {
    gcnn_keys.CELL: lambda n: np.repeat(np.eye(3, dtype=np.float32)[None], n, axis=0),
    # Open-boundary molecule: no dimension is periodic.  An all-False mask is the
    # neutral default, consistent with the identity-cell default above -- the
    # neighbour finder applies no cell transform on axes it sees as aperiodic.
    gcnn_keys.PBC: lambda n: np.zeros((n, 3), dtype=np.bool_),
    # Experimental stress (target), ``[n_graphs, 3, 3]`` in Cartesian form (the
    # Voigt 6-vector is normalised to 3x3 by ``gcnn.atomic.from_ase``).  A
    # molecule (no cell) has no well-defined stress, so a zero tensor is the
    # neutral default -- consistent with the identity-cell default above.
    atomic_keys.STRESS: lambda n: np.zeros((n, 3, 3), dtype=np.float32),
}


def _graph_keys(graph: jraph.GraphsTuple) -> set[str]:
    return (
        set(cast("dict[str, Any]", graph.nodes))
        | set(cast("dict[str, Any]", graph.edges))
        | set(cast("dict[str, Any]", graph.globals))
    )


def _normalize_graph_keys(graph: jraph.GraphsTuple, reference: set[str]) -> jraph.GraphsTuple:
    """Return *graph* with any ``reference`` key missing from it filled in.

    Only keys registered in ``_OPTIONAL_EDGE_KEYS`` / ``_OPTIONAL_GLOBAL_KEYS``
    may be missing (they are filled with a sensible default).  Any other missing
    key is a hard error -- callers should not be able to silently drop arbitrary
    node/edge/global features.
    """
    missing = reference - _graph_keys(graph)
    edge_updates: dict[str, np.ndarray] = {}
    global_updates: dict[str, np.ndarray] = {}
    unknown: set[str] = set()

    n_edge = int(np.asarray(graph.n_edge).sum())
    n_graph = int(np.asarray(graph.n_node).shape[0])
    for key in missing:
        if key in _OPTIONAL_EDGE_KEYS:
            edge_updates[key] = _OPTIONAL_EDGE_KEYS[key](n_edge)
        elif key in _OPTIONAL_GLOBAL_KEYS:
            global_updates[key] = _OPTIONAL_GLOBAL_KEYS[key](n_graph)
        else:
            unknown.add(key)

    if unknown:
        raise ValueError(
            f"Graph key-set mismatch: this graph is missing keys {sorted(unknown)} which "
            f"are present in other graphs in the dataset.  Either add them to this graph "
            f"or register them as optional fillable keys in "
            f"``_OPTIONAL_EDGE_KEYS`` / ``_OPTIONAL_GLOBAL_KEYS`` in this module."
        )

    return graph._replace(
        edges=dict(graph.edges, **edge_updates),
        globals=dict(graph.globals, **global_updates),
    )


def _normalize_graphs(
    graphs: Sequence[jraph.GraphsTuple], reference: set[str]
) -> Sequence[jraph.GraphsTuple]:
    """Return *graphs* with each graph's key set reconciled to *reference*.

    *reference* is the union of keys present across the **whole** dataset
    (every split), not just this one -- that is what lets a molecule-only split
    and a crystal-only split settle onto a single shared schema.  Missing keys
    that are
    registered in ``_OPTIONAL_EDGE_KEYS`` / ``_OPTIONAL_GLOBAL_KEYS`` are filled
    with a neutral default (identity cell, zero edge-cell-shifts, zero stress)
    so an open-boundary molecule can sit in the same batch as a periodic crystal.
    Any *other* missing key raises :class:`ValueError` so the user is forced to
    reconcile it rather than silently dropping a real feature.
    """
    needs_update = {i for i, graph in enumerate(graphs) if _graph_keys(graph) != reference}
    if not needs_update:
        # Already homogeneous against *reference*: return the original sequence
        # untouched (avoids needlessly rebuilding every GraphsTuple in the common
        # case where the dataset is already coherent).  Also covers empty input.
        return graphs

    updated = {i: _normalize_graph_keys(graphs[i], reference) for i in needs_update}
    return [updated.get(i, g) for i, g in enumerate(graphs)]


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

        (
            self.data_train,
            self.data_val,
            self.data_test,
            self._max_padding,
        ) = self._reconcile_splits()
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

    def _reconcile_splits(
        self,
    ) -> tuple[
        GraphDataset | None,
        GraphDataset | None,
        GraphDataset | None,
        "gcnn.data.GraphPadding | None",
    ]:
        """Reconcile the key sets across every split and compute the shared padding.

        Two passes, in the only correct order:

        1. **Reference key set** is computed *across all splits* (not per split),
           so a molecule-only train split and a crystal-only val split settle onto
           one shared schema.  Normalization is "fill in keys present somewhere
           in the dataset", which requires this global reference to be known first
           -- hence normalization cannot be fused into this pass or made
           order-dependent.
        2. **Normalize + pad** each split against that one reference.  Padding is
           derived from the *normalized* graphs (the filled-in ``cell`` /
           ``stress`` arrays are part of what the padding sees), so the
           normalize-then-pad dependency is encoded here rather than left to the
           line order of ``setup``.

        Called only from ``setup()`` after the "all-None" guard has raised, so at
        least one split is non-``None``.  The shared padding is the elementwise
        max over the per-split paddings.
        """
        splits = {
            "train": self.data_train,
            "val": self.data_val,
            "test": self.data_test,
        }

        reference: set[str] = set()
        for ds in splits.values():
            # An ``is not None`` check (not truthiness) so an *empty* split is
            # iterated here: it contributes no keys, but is still normalised and
            # padded in the second pass.  Deliberately asymmetric with the
            # ``is None: continue`` guard below -- both are the same intent.
            if ds is not None:
                for graph in ds:
                    reference |= _graph_keys(graph)

        batch_size = self._batch_size if self._batch_mode is _common.BatchMode.IMPLICIT else 1
        pad_to_multiple = (
            self._pad_to_multiple if self._batch_mode is _common.BatchMode.IMPLICIT else None
        )

        normalized: dict[str, GraphDataset | None] = {}
        paddings: "list[gcnn.data.GraphPadding]" = []
        for name, ds in splits.items():
            if ds is None:
                normalized[name] = None
                continue
            ds = _normalize_graphs(ds, reference)
            normalized[name] = ds
            paddings.append(
                _batching.GraphBatcher.calculate_padding(
                    ds,
                    batch_size,
                    pad_to_multiple=pad_to_multiple,
                )
            )

        shared = _batching.max_padding(*paddings) if paddings else None
        return normalized["train"], normalized["val"], normalized["test"], shared

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
