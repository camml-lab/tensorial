import abc
from typing import Generic, TypeVar

from typing_extensions import override

D = TypeVar("D")


class DataFetcher(Generic[D], abc.ABC):
    """Abstract base class for data fetchers."""

    @abc.abstractmethod
    def fetch(self) -> D:
        """Fetch data from a source."""


class PassthroughFetcher(DataFetcher[D]):
    def __init__(self, dataset: D):
        self._dataset = dataset

    @override
    def fetch(self) -> D:
        return self._dataset
