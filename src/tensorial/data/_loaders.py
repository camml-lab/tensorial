# -*- coding: utf-8 -*-
import collections.abc
from typing import Iterator, Tuple

import jax

from . import _types, samplers

__all__ = ("ArrayLoader", "CachingLoader")


class ArrayLoader(collections.abc.Iterable[Tuple[jax.typing.ArrayLike, ...]]):
    """A dataset of arrays"""

    def __init__(
        self,
        *arrays: jax.typing.ArrayLike,
        batch_size: int = 1,
        shuffle=False,
    ):
        if not all(arrays[0].shape[0] == array.shape[0] for array in arrays):
            raise ValueError("Size mismatch between tensors")

        self._arrays = arrays
        self._sampler = samplers.create_sequence_sampler(
            arrays[0], batch_size=batch_size, shuffle=shuffle
        )

    def __iter__(self) -> Iterator[Tuple[jax.typing.ArrayLike, ...]]:
        for idx in self._sampler:
            yield tuple(array[idx] for array in self._arrays)

    def __len__(self) -> int:
        return len(self._sampler)


class CachingLoader(collections.abc.Iterable):
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
        # Check if we should clear the cache
        if self._time_since_reset == self._reset_every:
            self._cache = []
            self._time_since_reset = 0
