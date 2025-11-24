from collections.abc import Sequence
from typing import TYPE_CHECKING, Final

import jraph
import reax
from typing_extensions import override

from . import _batching, _dataloader

if TYPE_CHECKING:
    from tensorial import gcnn

__all__ = ("GraphDataModule",)

Dataset = Sequence[jraph.GraphsTuple]


class GraphDataModule(reax.DataModule):
    """A data module that serves jraph.GraphsTuples"""

    def __init__(
        self,
        dataset: Sequence[jraph.GraphsTuple],
        train_val_test_split: Sequence[int | float] = (0.85, 0.05, 0.1),
        batch_size: int = 32,
    ):
        """Initialize the module

        Args:
            dataset: The data loader of all the graphs to use
            train_val_test_split: The train, validation, and test split.
            batch_size: The batch size. Defaults to `32`.
        """
        super().__init__()
        # Params
        self._train_val_test_split: Final[tuple[int | float, ...]] = tuple(train_val_test_split)
        self._batch_size: Final[int] = batch_size

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
            # Split up the data
            train, val, test = reax.data.random_split(
                stage.rngs,
                dataset=self._dataloader,
                lengths=self._train_val_test_split,
            )
            graph_datasets: dict[str, Dataset] = dict(train=train, val=val, test=test)

            # Calculate the maximum padding to use
            paddings: "list[gcnn.data.GraphPadding]" = []
            for graphs in graph_datasets.values():
                paddings.append(_batching.GraphBatcher.calculate_padding(graphs, self._batch_size))

            self.data_train = graph_datasets["train"]
            self.data_val = graph_datasets["val"]
            self.data_val = graph_datasets["val"]
            self.data_test = graph_datasets["test"]

            # Calculate a padding that will work for all the datasets.
            self._max_padding = _batching.max_padding(*paddings)

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
        )
