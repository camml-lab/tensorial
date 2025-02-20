from collections.abc import Iterable, Iterator
from typing import TypeVar, Union

import beartype
import jax
import jaxtyping as jt
import numpy as np

from . import _types, samplers

__all__ = ("ArrayLoader", "CachingLoader")


T = TypeVar("T")


def _single_or_value(value: tuple[T, ...], to_test=None) -> Union[T, tuple[T, ...]]:
    if to_test is None:
        to_test = value
    if len(to_test) > 1:
        return tuple(value)
    return value[0]


ArrayOrArrayTuple = Union[jax.typing.ArrayLike, tuple[jax.typing.ArrayLike, ...]]


class ArrayLoader(Iterable[ArrayOrArrayTuple]):
    """A dataset of arrays"""

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        *arrays: jax.typing.ArrayLike,
        batch_size: int = 1,
        shuffle=False,
    ):
        if not all(arrays[0].shape[0] == array.shape[0] for array in arrays):
            raise ValueError("Size mismatch between tensors")

        self._arrays = arrays
        self._sampler: _types.Sampler[list[int]] = samplers.create_sequence_sampler(
            arrays[0], batch_size=batch_size, shuffle=shuffle
        )

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __iter__(self) -> Iterator[ArrayOrArrayTuple]:
        for idx in self._sampler:
            value = tuple(array.take(np.array(idx), axis=0) for array in self._arrays)
            yield _single_or_value(value, self._arrays)

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __len__(self) -> int:
        return len(self._sampler)

    def first(self) -> ArrayOrArrayTuple:
        return next(iter(self))


class CachingLoader(Iterable):
    """
    Caching loader is useful, for example, if you don't want to shuffle data every time but at
    some interval defined by ``repeat_every``.  Naturally, this means you need to have enough memory
    to accommodate all the data.

    """

    def __init__(self, loader: _types.DataLoader, reset_every: int):
        self._loader = loader
        self._reset_every = reset_every
        self._time_since_reset = 0
        self._cache = None

    def __iter__(self):
        if self._cache:
            yield from self._cache
        else:
            # Have to pull from the loader
            cache = []
            for entry in self._loader:
                yield entry
                cache.append(entry)
            self._cache = cache

        self._time_since_reset += 1
        # Check if we should clear the cache for the next iteration
        if self._time_since_reset == self._reset_every:
            self._cache = []
            self._time_since_reset = 0

    def __len__(self):
        # Get it from the cache
        if self._time_since_reset > 1:
            return len(self._cache)

        # otherwise, from the loader, but the loader may not be `Sized` which will cause TypeError
        return len(self._loader)
