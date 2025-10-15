from collections.abc import Iterable, Sequence
from typing import TypeVar, Union

__all__ = "Sampler", "DataLoader", "Dataset"

T_co = TypeVar("T_co", covariant=True)
IdxT = TypeVar("IdxT")

Dataset = Union[Iterable[T_co], Sequence[T_co]]
DataLoader = Iterable[T_co]
Sampler = Iterable[IdxT]
