import jaxtyping as jt

from tensorial.typing import Array

from .. import utils

__all__ = ("cell_volume",)


def cell_volume(cell_vectors: jt.Float[Array, "3 3"], np_=None) -> Array:
    """
    Computes the volume of a unit cell defined by its cell vectors.

    The cell volume is calculated as the absolute value of the determinant of the
    matrix formed by the cell vectors. This is commonly used in crystallography
    and computational physics to determine the volume of a unit cell.

    Args:
        cell_vectors: A 3x3 matrix where each row represents a cell vector in
            three-dimensional space.
        np_: The numerical library backend to use for computation. If None, the
            backend is inferred from the input tensor.

    Returns:
        The volume of the unit cell as a scalar tensor.

    Raises:
        LinAlgError: If the cell vectors are linearly dependent, resulting in a
            zero determinant.
    """
    if np_ is None:
        np_ = utils.infer_backend(cell_vectors)

    return np_.abs(np_.linalg.det(cell_vectors))
