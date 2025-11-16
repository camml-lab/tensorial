from collections.abc import Sequence
from typing import Annotated, Generic, TypeVar, Union

import e3nn_jax as e3j
import jax.typing
import jaxtyping as jt
import numpy as np

__all__ = (
    "ArrayType",
    "IrrepLike",
    "IrrepsLike",
    "IntoIrreps",
    "IrrepsArrayShape",
    "IndexArray",
    "CellType",
    "PbcType",
)


# pylint: disable=invalid-name

ArrayT = TypeVar("ArrayT")
ValueT = TypeVar("ValueT")


class _Helper(Generic[ArrayT, ValueT]):
    def __init__(self, array_type: ArrayT, value_type: ValueT):
        self._array_type = array_type
        self._value_type = value_type

    def __getitem__(self, shape: str) -> "ValueT[ArrayT]":
        return self._value_type[self._array_type, shape]


IrrepLike = str | e3j.Irrep
IrrepsLike = str | e3j.Irreps | tuple[e3j.MulIrrep]
IntoIrreps = Union[
    None,
    e3j.Irrep,
    e3j.MulIrrep,
    str,
    e3j.Irreps,
    Sequence[str | e3j.Irrep | e3j.MulIrrep | tuple[int, "IntoIrreps"]],
]
# "IrrepsArray with explicit shape"
IrrepsArrayShape = _Helper(e3j.IrrepsArray, jt.Float)


Array = Union[jax.Array, np.ndarray]
ArrayType = Union[jax.Array, np.ndarray]
# "Integer array that is used as an index"
IndexArray = _Helper(Array, jt.Int)  # pylint: disable=invalid-name
CellType = Annotated[
    jt.Float[ArrayType, "3 3"], "Unit cell array i.e. rows containing cell vectors"
]
PbcType = Annotated[
    tuple[bool, bool, bool] | jt.Bool[jax.typing.ArrayLike, "3"],
    "Boolean sequence indicating whether boundaries are periodic in each a, b, c directions",
]
