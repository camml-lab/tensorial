from typing import TYPE_CHECKING, Union

import jax
import jaxtyping as jt
import numpy as np

from .. import utils

if TYPE_CHECKING:
    from tensorial import typing


__all__ = ("cell_volume",)


def cell_volume(
    cell_vectors: "jt.Float[typing.ArrayType, '3 3']", np_=None
) -> Union[np.ndarray, jax.Array]:
    if np_ is None:
        np_ = utils.infer_backend(cell_vectors)

    return np_.abs(np_.linalg.det(cell_vectors))
