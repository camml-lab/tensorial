# -*- coding: utf-8 -*-
from typing import Generic, Tuple, TypeVar, Union

import e3nn_jax as e3j
import jax.typing
import jaxtyping as jt
import numpy as np

IrrepLike = Union[str, e3j.Irrep]
IrrepsLike = Union[str, e3j.Irreps, Tuple[e3j.MulIrrep]]

ArrayT = TypeVar("ArrayT")
ValueT = TypeVar("ValueT")


class _Helper(Generic[ArrayT, ValueT]):
    def __init__(self, array_type: ArrayT, value_type: ValueT):
        self._array_type = array_type
        self._value_type = value_type

    def __getitem__(self, shape: str) -> "ValueT[ArrayT]":
        return self._value_type[self._array_type, shape]


ArrayType = Union[jax.Array, np.ndarray]
IrrepsArrayShape = _Helper(e3j.IrrepsArray, jt.Float)
IndexArray = _Helper(ArrayType, jt.Int)
