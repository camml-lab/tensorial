import abc
from typing import Generic, TypeVar

import equinox

__all__ = ("Metric",)

OutT = TypeVar("OutT")


class Metric(equinox.Module, Generic[OutT], metaclass=abc.ABCMeta):
    Self = TypeVar("Self", bound="Metric")

    @classmethod
    def empty(cls) -> Self:
        """Create a new empty instance.

        By default, this will call the constructor with no arguments, if needed, subclasses can
        overwrite this with custom behaviour.
        """
        return cls()

    @classmethod
    @abc.abstractmethod
    def create(cls, *args, **kwargs) -> Self:
        """Create the metric from data"""

    def update(self, *args, **kwargs) -> "Metric":
        """Update the metric from new data and return a new instance"""
        return self.merge(self.create(*args, **kwargs))

    @abc.abstractmethod
    def merge(self, other: "Metric") -> "Metric":
        """Merge the metric with data from another metric instance of the same type"""

    @abc.abstractmethod
    def compute(self) -> OutT:
        """Compute the metric"""
