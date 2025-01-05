import math
from typing import Optional

import beartype
import jax.typing
import jaxtyping as jt
import numpy as np

from tensorial import typing

from . import distances


def get_cell_multiple_range(
    cell: jax.typing.ArrayLike, cell_vector: int, cutoff: float
) -> tuple[int, int]:
    multiplier = get_max_cell_vector_repetitions(cell, cell_vector, cutoff=cutoff)
    return -math.ceil(multiplier), math.ceil(multiplier) + 1


def get_cell_multiple_ranges(
    cell: typing.CellType,
    cutoff: float,
    pbc: Optional[typing.PbcType] = (True, True, True),
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    return tuple(
        (get_cell_multiple_range(cell, cell_vector, cutoff=cutoff) if pbc[cell_vector] else (0, 1))
        for cell_vector in (0, 1, 2)
    )


def get_max_cell_vector_repetitions(
    cell: typing.CellType, cell_vector: int, cutoff: float
) -> float:
    """
    Given a unit cell defined by three vectors this will return the number of multiples of the
    vector indexed by `cell_vector` that are needed to reach the edge of a sphere with radius
    ``cutoff``. This tells you what multiple of cell vectors you need to go up to (when rounded up
    to the nearest integer) in order to fully cover all points in the sphere, in teh given cell
    vector direction.
    """
    cell = np.asarray(cell)
    vec1 = (cell_vector + 1) % 3
    vec2 = (cell_vector + 2) % 3
    volume = cell_volume(cell).item()

    vec1_cross_vec2_len = np.linalg.norm(np.cross(cell[vec1], cell[vec2])).item()
    return get_num_plane_repetitions_to_bound_sphere(cutoff, volume, vec1_cross_vec2_len)


def get_num_plane_repetitions_to_bound_sphere(
    radius: float, volume: float, cross_len: float
) -> float:
    # The vector normal to the plane
    return radius / volume * cross_len


def cell_volume(cell: typing.CellType) -> jax.Array:
    return np.abs(np.dot(cell[0], np.cross(cell[1], cell[2])))


def sphere_volume(radius: float) -> float:
    return (4.0 / 3.0) * np.pi * radius**3


@jt.jaxtyped(typechecker=beartype.beartype)
def get_edge_vectors(
    positions: jax.typing.ArrayLike, edges: distances.Edges, cell: typing.CellType
) -> jax.typing.ArrayLike:
    return positions[edges.to_idx] - positions[edges.from_idx] + (edges.cell_shift @ cell)
