"""Low-level unit-cell geometry utilities: volumes and lattice image counting."""

import math

import beartype
import jax.typing
import jaxtyping as jt
import numpy as np

from tensorial.typing import CellType, PbcType

from . import distances


def get_cell_multiple_range(cell: jt.ArrayLike, cell_vector: int, cutoff: float) -> tuple[int, int]:
    """Return the integer range of cell-image multiples needed along one cell direction.

    Args:
        cell: the unit cell (rows are the three cell vectors).
        cell_vector: which cell vector to query along (0, 1 or 2).
        cutoff: the search radius.

    Returns:
        a half-open ``(start, stop)`` range of multiples covering the cutoff sphere.
    """
    multiplier = get_max_cell_vector_repetitions(cell, cell_vector, cutoff=cutoff)
    return -math.ceil(multiplier), math.ceil(multiplier) + 1


def get_cell_multiple_ranges(
    cell: CellType, cutoff: float, pbc: PbcType | None = (True, True, True)
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Return per-cell-vector image ranges covering the cutoff sphere.

    Non-periodic directions simply return ``(0, 1)`` (no images).
    """
    return tuple(
        (get_cell_multiple_range(cell, cell_vector, cutoff=cutoff) if pbc[cell_vector] else (0, 1))
        for cell_vector in (0, 1, 2)
    )


def get_max_cell_vector_repetitions(cell: CellType, cell_vector: int, cutoff: float) -> float:
    """Number of multiples of one cell vector that are needed to reach the edge of a sphere of
    radius ``cutoff``.  This tells you how far along the given cell-vector direction you need to go
    (rounded up to the nearest integer) in order to fully cover all points in the sphere.
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
    """Number of cell-plane repetitions needed to bound a sphere of the given radius.

    Derived from ``radius / volume * cross_len`` (distance to the next parallel plane).
    """
    # The vector normal to the plane
    return radius / volume * cross_len


def cell_volume(cell: CellType) -> jax.Array:
    """The (scalar) volume of a unit cell given by its three cell vectors."""
    return np.abs(np.dot(cell[0], np.cross(cell[1], cell[2])))


def sphere_volume(radius: float) -> float:
    """Volume of a sphere of the given radius."""
    return (4.0 / 3.0) * np.pi * radius**3


@jt.jaxtyped(typechecker=beartype.beartype)
def get_edge_vectors(
    positions: jt.ArrayLike, edges: distances.Edges, cell: CellType
) -> jt.ArrayLike:
    """Compute the (periodic-corrected) edge vectors for a set of edges.

    Args:
        positions: the particle positions, shape ``(N, 3)``.
        edges: the neighbour edges, with per-edge cell-image shifts.
        cell: the unit cell used to convert cell-image shifts into Cartesian deltas.

    Returns:
        the edge vectors, shape ``(E, 3)``.
    """
    return positions[edges.to_idx] - positions[edges.from_idx] + (edges.cell_shift @ cell)
