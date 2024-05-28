# -*- coding: utf-8 -*-
from typing import Generic, Tuple, TypeVar, Union

import e3nn_jax as e3j
import jax.typing
import jaxtyping as jt
import numpy as np

IrrepLike = Union[str, e3j.Irrep]
IrrepsLike = Union[str, e3j.Irreps, Tuple[e3j.MulIrrep]]

T = TypeVar("T")


class _Helper(Generic[T]):
    def __init__(self, typ: T):
        self._type = typ


class _FloatArray(_Helper[T]):
    def __getitem__(self, shape: str):
        return jt.Float[self._type, shape]


class _IntArray(_Helper[T]):
    def __getitem__(self, shape: str):
        return jt.Int[self._type, shape]


ArrayType = Union[jax.Array, np.ndarray]
IrrepsArrayShape = _FloatArray(e3j.IrrepsArray)
IndexArray = _IntArray(ArrayType)
