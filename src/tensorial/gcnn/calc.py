import jaxtyping as jt

from tensorial.typing import Array

from .. import utils

__all__ = ("cell_volume",)


def cell_volume(cell_vectors: jt.Float[Array, "3 3"], np_=None) -> Array:
    if np_ is None:
        np_ = utils.infer_backend(cell_vectors)

    return np_.abs(np_.linalg.det(cell_vectors))
