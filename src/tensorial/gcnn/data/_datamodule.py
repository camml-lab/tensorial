from collections.abc import Sequence
import logging
from typing import TYPE_CHECKING, Final

import jraph
import numpy as np
import reax
from typing_extensions import override

from . import _batching, _common, _dataloader

if TYPE_CHECKING:
    from tensorial import gcnn

__all__ = ("GraphDataModule",)

Dataset = Sequence[jraph.GraphsTuple]

_LOGGER = logging.getLogger(__name__)


class GraphDataModule(reax.DataModule):
    """A data module that serves jraph.GraphsTuples"""

    def __init__(
        self,
        dataset: Sequence[jraph.GraphsTuple],
        train_val_test_split: Sequence[int | float] = (0.85, 0.05, 0.1),
        batch_size: int = 32,
        batch_mode: "gcnn.data.BatchMode | str" = _common.BatchMode.IMPLICIT,
        # --- NEW: k-fold cross-validation support -----------------------------------------
        # These three parameters are additive and fully optional. When `n_folds` is left at
        # its default value of 1, the datamodule behaves EXACTLY as before: a single random
        # split governed by `train_val_test_split`. This preserves backward compatibility for
        # every existing config (e.g. the BEC/nequip_electric models) that instantiates this
        # class without knowledge of k-fold.
        #
        # K-fold is only activated when a caller explicitly sets `n_folds > 1` (and provides
        # `test_fold`), which is currently the case for the Pockels tensor study, where the
        # dataset is small (~67 structures) and a single fixed 80/10/10 split leaves a test set
        # too small to produce a statistically meaningful parity plot.
        test_fold: int | None = None,
        n_folds: int = 1,
        fold_seed: int = 42,
        # ------------------------------------------------------------------------------------
    ):
        """Initialize the module

        Args:
            dataset: The data loader of all the graphs to use
            train_val_test_split: The train, validation, and test split. When k-fold is active
                (`n_folds > 1`), only the first two entries are used (as relative weights for
                the train/val split of the folds not held out for testing); the third entry
                is ignored, since the test set is instead determined by `test_fold`/`n_folds`.
            batch_size: The batch size. Defaults to `32`.
            test_fold: Index of the fold (in `[0, n_folds)`) to hold out as the test set for
                this run. Required when `n_folds > 1`. Ignored when `n_folds == 1`.
            n_folds: Number of folds to partition `dataset` into for k-fold cross-validation.
                Defaults to `1`, which disables k-fold entirely and falls back to the original
                single random train/val/test split behaviour.
            fold_seed: Fixed seed used ONLY to determine the k-fold partition (i.e. which
                sample ends up in which fold). This is intentionally separate from the
                trainer/experiment seed (`cfg.seed`) so that the fold assignment is stable and
                reproducible across runs/experiments regardless of how the general training
                seed is configured -- otherwise "fold 0" could silently refer to a different
                subset of structures from one run to the next, which would break the
                out-of-fold aggregation used to build a combined parity plot.
        """
        super().__init__()
        # Params
        self._train_val_test_split: Final[tuple[int | float, ...]] = tuple(train_val_test_split)
        self._batch_size: Final[int] = batch_size
        self._batch_mode: "Final[gcnn.data.BatchMode]" = _common.BatchMode(batch_mode)
        # --- NEW: k-fold params (see docstring above) ---
        self._test_fold: Final[int | None] = test_fold
        self._n_folds: Final[int] = n_folds
        self._fold_seed: Final[int] = fold_seed
        # -------------------------------------------------
        # State
        self._dataloader = dataset
        self.batch_size_per_device = batch_size
        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None
        self._max_padding: "gcnn.data.GraphPadding | None" = None

    @override
    def setup(self, stage: "reax.Stage", /) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by REAX before `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`, so be careful not to execute things like random
        split twice! Also, it is called after `self.prepare_data()` and there is a barrier in
        between which ensures that all the processes proceed to `self.setup()` once the data is
        prepared and available for use.

        Args:
            stage: The stage to setup. Either `"fit"`, `"validate"`,
                `"test"`, or `"predict"`.
        Defaults to ``None``.
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:

            # --- NEW: branch between k-fold splitting and the original random split ---
            if self._n_folds > 1:
                train, val, test = self._kfold_split()
            else:
                # Original behaviour, unchanged: a single random split of the whole dataset.
                train, val, test = reax.data.random_split(
                    self.rngs,
                    dataset=self._dataloader,
                    lengths=self._train_val_test_split,
                )
            # ---------------------------------------------------------------------------

            graph_datasets: dict[str, Dataset] = dict(train=train, val=val, test=test)
            # Calculate the maximum padding to use
            paddings: "list[gcnn.data.GraphPadding]" = []
            # Padding is computed with _batching.max_padding(*paddings) -- few lines below.
            # In explicit mode, if we used self._batch_size to compute the padding,
            # the number of padding graphs would follow the implicit batching logic.
            # Thus, in explicit mode we would obtain for the resulting batch:
            #     batch.n_node.shape == (self._batch_size, self._batch_size + 1)
            # With this condition, we instead enforce:
            #     batch.n_node.shape == (self._batch_size, 2)
            # which is the expected behavior for explicit batching.
            for graphs in graph_datasets.values():
                if self._batch_mode is _common.BatchMode.IMPLICIT:
                    paddings.append(
                        _batching.GraphBatcher.calculate_padding(graphs, self._batch_size)
                    )
                else:
                    paddings.append(_batching.GraphBatcher.calculate_padding(graphs, 1))
            self.data_train = graph_datasets["train"]
            self.data_val = graph_datasets["val"]
            self.data_test = graph_datasets["test"]
            # Calculate a padding that will work for all the datasets.
            self._max_padding = _batching.max_padding(*paddings)

    # --- NEW: helper implementing the k-fold partition logic ---
    def _kfold_split(self) -> tuple[Dataset, Dataset, Dataset]:
        """Partition `self._dataloader` into k folds and return (train, val, test).

        The fold assignment is deterministic and based solely on `self._fold_seed` and
        `self._n_folds` -- NOT on `self.rngs` -- so that the same sample always lands in the
        same fold across every run of the k-fold procedure (run 0..n_folds-1), regardless of
        the general training seed. This is what guarantees that:
          - the k folds are disjoint and together cover the entire dataset exactly once, and
          - concatenating the test-set predictions from all n_folds runs later on yields one
            prediction per sample in the full dataset, with no duplicates and no leakage
            (every sample was held out exactly once, by the model trained without it).

        The train/val split *within* the folds not held out as test is still randomised using
        `self.rngs`, exactly like the original single-split behaviour -- only the assignment
        of the test fold itself needs to be stable across runs.
        """
        if self._test_fold is None:
            raise ValueError(
                "`test_fold` must be provided when `n_folds` > 1 (got test_fold=None)."
            )
        if not 0 <= self._test_fold < self._n_folds:
            raise ValueError(
                f"`test_fold`={self._test_fold} is out of range for n_folds={self._n_folds} "
                f"(expected 0 <= test_fold < n_folds)."
            )

        n_samples = len(self._dataloader)
        # Fixed, dedicated RNG for fold assignment -- decoupled from the experiment seed.
        fold_rng = np.random.default_rng(self._fold_seed)
        shuffled_indices = fold_rng.permutation(n_samples)
        folds = np.array_split(shuffled_indices, self._n_folds)

        test_indices = folds[self._test_fold]
        train_val_indices = np.concatenate(
            [folds[i] for i in range(self._n_folds) if i != self._test_fold]
        )

        test_data = [self._dataloader[i] for i in test_indices]
        train_val_data = [self._dataloader[i] for i in train_val_indices]

        # Re-normalise the first two entries of train_val_test_split as train/val weights,
        # since the third (test) entry is superseded by the fold-based test set above.
        train_weight, val_weight = self._train_val_test_split[0], self._train_val_test_split[1]
        weight_total = train_weight + val_weight
        train_data, val_data = reax.data.random_split(
            self.rngs,
            dataset=train_val_data,
            lengths=(train_weight / weight_total, val_weight / weight_total),
        )

        _LOGGER.info(
            "K-fold split active: test_fold=%d/%d -> %d train, %d val, %d test " "(fold_seed=%d)",
            self._test_fold,
            self._n_folds,
            len(train_data),
            len(val_data),
            len(test_data),
            self._fold_seed,
        )

        return train_data, val_data, test_data

    # ---------------------------------------------------------------

    @override
    def train_dataloader(self) -> reax.DataLoader:
        """Create and return the train dataloader.

        Returns:
            The train dataloader.
        """
        if self.data_train is None:
            raise reax.exceptions.MisconfigurationException(
                "Must call setup() before requesting the dataloader"
            )
        return _dataloader.GraphLoader(
            self.data_train,
            batch_size=self._batch_size,
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
        if self.data_val is None:
            raise reax.exceptions.MisconfigurationException(
                "Must call setup() before requesting the dataloader"
            )
        return _dataloader.GraphLoader(
            self.data_val,
            batch_size=self.batch_size_per_device,
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
        if self.data_test is None:
            raise reax.exceptions.MisconfigurationException(
                "Must call setup() before requesting the dataloader"
            )
        return _dataloader.GraphLoader(
            self.data_test,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            padding=self._max_padding,
            pad=True,
            batch_mode=self._batch_mode,
        )
