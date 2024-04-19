# -*- coding: utf-8 -*-
import collections
import functools
import math
from typing import Tuple

import equinox
import jax
import jax.numpy as jnp

Edges = collections.namedtuple('Edge', 'from_idx to_idx cell_shift')
i32 = jnp.int32  # pylint: disable=invalid-name

DEFAULT_MAX_CELL_MULTIPLES = 10_000
MASK_VALUE = -1


def get_num_plane_repetitions_to_bound_sphere(radius: float, volume: float, cross_len: float) -> float:
    # The vector normal to the plane
    return radius / volume * cross_len


def cell_volume(cell: jax.Array) -> jax.Array:
    return jnp.abs(jnp.dot(cell[0], jnp.cross(cell[1], cell[2])))


def sphere_volume(radius: float) -> float:
    return (4 / 3) * jnp.pi * radius**3


class Periodic(equinox.Module):
    _cell: jax.Array
    _cutoff: float
    _max_cell_multiples: int
    _a: jax.Array
    _b: jax.Array
    _c: jax.Array
    _volume: float

    _a_cross_b: jax.Array
    _a_cross_b_len: float
    _a_cross_b_hat: jax.Array

    _b_cross_c: jax.Array
    _b_cross_c_len: float
    _b_cross_c_hat: jax.Array

    _a_cross_c: jax.Array
    _a_cross_c_len: float
    _a_cross_c_hat: jax.Array

    def __init__(self, cell: jax.Array, cutoff: float, max_cell_multiples=100_000):
        self._cell = cell
        self._cutoff = cutoff
        self._max_cell_multiples = max_cell_multiples

        self._a = self._cell[0]
        self._b = self._cell[1]
        self._c = self._cell[2]
        self._volume = jnp.abs(jnp.dot(self._a, jnp.cross(self._b, self._c)))

        self._a_cross_b = jnp.cross(self._a, self._b)
        self._a_cross_b_len = jnp.linalg.norm(self._a_cross_b)
        self._a_cross_b_hat = self._a_cross_b / self._a_cross_b_len

        self._b_cross_c = jnp.cross(self._b, self._c)
        self._b_cross_c_len = jnp.linalg.norm(self._b_cross_c)
        self._b_cross_c_hat = self._b_cross_c / self._b_cross_c_len

        self._a_cross_c = jnp.cross(self._a, self._c)
        self._a_cross_c_len = jnp.linalg.norm(self._a_cross_c)
        self._a_cross_c_hat = self._a_cross_c / self._a_cross_c_len

    def __call__(self, r1: jax.Array, r2: jax.Array):
        cutoff = self._cutoff
        vol = self._volume
        # TODO: Wrap a, and b into the current unit cell
        dr = r2 - r1

        a_max = math.floor(
            get_num_plane_repetitions_to_bound_sphere(
                cutoff + jnp.fabs(jnp.dot(dr, self._b_cross_c_hat)),
                vol,
                self._b_cross_c_len,
            )
        )

        b_max = math.floor(
            get_num_plane_repetitions_to_bound_sphere(
                cutoff + jnp.fabs(jnp.dot(dr, self._a_cross_c_hat)),
                vol,
                self._a_cross_c_len,
            )
        )

        c_max = math.floor(
            get_num_plane_repetitions_to_bound_sphere(
                cutoff + jnp.fabs(jnp.dot(dr, self._a_cross_b_hat)),
                vol,
                self._a_cross_b_len,
            )
        )

        a_max = min(a_max, self._max_cell_multiples)
        b_max = min(b_max, self._max_cell_multiples)
        c_max = min(c_max, self._max_cell_multiples)

        cutoff_sq = cutoff * cutoff
        vectors = []
        cell_indices = []

        for i in range(-a_max, a_max + 1):
            ra = i * self._a
            for j in range(-b_max, b_max + 1):
                rab = ra + j * self._b
                for k in range(-c_max, c_max + 1):
                    displacement = rab + k * self._c
                    out_vec = displacement + dr
                    if jnp.dot(out_vec, out_vec) < cutoff_sq:
                        vectors.append(out_vec)
                        cell_indices.append(jnp.array([i, j, k]))

        if not vectors:
            return jnp.zeros((0, 3)), jnp.zeros((0, 3), dtype=int)

        return jnp.vstack(vectors), jnp.vstack(cell_indices)


class NeighbourList(equinox.Module):
    neighbours: jax.Array
    cell_indices: jax.Array
    actual_max_neighbours: int
    _finder: 'NeighbourFinder'

    def __init__(
        self,
        neighbours: jax.typing.ArrayLike,
        cell_indices: jax.typing.ArrayLike,
        actual_max_neighbours: jax.Array = -1,
        finder: 'NeighbourFinder' = None,
    ):
        # if neighbours.shape != cell_indices.shape[:2]:
        #     raise ValueError("Cell indices and neighbours must have same shape")
        # if jnp.any(neighbours > neighbours.shape[0]):
        #     raise ValueError(
        #         "One or more entries in the neighbours array refers to an index higher than the maximum possible")

        self.neighbours = jnp.asarray(neighbours)
        self.cell_indices = jnp.asarray(cell_indices)
        self.actual_max_neighbours = actual_max_neighbours
        self._finder = finder

    @property
    def num_particles(self) -> int:
        return self.neighbours.shape[0]

    @property
    def max_neighbours(self) -> int:
        return self.neighbours.shape[1]

    @property
    def did_overflow(self) -> bool:
        """Returns `True` if the list could not accommodate all the neighbours.  The actual number needed is stored in
        `actual_max_neighbours`"""
        return self.actual_max_neighbours > self.max_neighbours

    def get_edges(self, self_interaction=False) -> Edges:
        mask = self.neighbours != MASK_VALUE
        if not self_interaction:
            mask = mask.at[:, 0].set(False)

        from_idx = jnp.repeat(jnp.arange(0, self.num_particles)[:, None], self.max_neighbours, axis=1)
        return Edges(from_idx[mask], self.neighbours[mask], self.cell_indices[mask])

    def list_overflow(self) -> bool:
        return self.actual_max_neighbours > self.max_neighbours

    def reallocate(self, positions: jax.typing.ArrayLike) -> 'NeighbourList':
        return self._finder.get_neighbours(positions, max_neighbours=self.actual_max_neighbours)


def get_edge_vectors(positions: jax.typing.ArrayLike, edges: Edges, cell: jax.typing.ArrayLike) -> jax.Array:
    return positions[edges.to_idx] + (edges.cell_shift @ cell) - positions[edges.from_idx]


class NeighbourFinder(equinox.Module):

    def get_neighbours(self, positions: jax.typing.ArrayLike, max_neighbours: int = None) -> NeighbourList:
        """Get the neighbour list for the given positions"""

    def estimate_neighbours(self, positions: jax.typing.ArrayLike) -> int:
        """Estimate the number of neighbours per particle"""


class OpenBoundary(NeighbourFinder):
    _cutoff: float

    def __init__(self, cutoff: float):
        self._cutoff = cutoff

    def get_neighbours(self, positions: jax.typing.ArrayLike, max_neighbours: int = None) -> NeighbourList:
        positions = jnp.asarray(positions)
        max_neighbours = max_neighbours or self.estimate_neighbours(positions)
        # Get the neighbours mask
        neigh_idx = jax.vmap(neighbours_mask_direct, (0, None, None))(positions, positions, self._cutoff)

        get_neighbours = functools.partial(jnp.argwhere, size=max_neighbours, fill_value=-1)
        to_idx = jax.vmap(get_neighbours)(neigh_idx)[..., 0]

        cell_indices = jnp.zeros((*to_idx.shape, 3), dtype=int)
        return NeighbourList(to_idx, cell_indices, actual_max_neighbours=jnp.max(neigh_idx.sum(axis=1)), finder=self)

    def estimate_neighbours(self, positions: jax.typing.ArrayLike) -> int:
        positions = jnp.asarray(positions)
        approx_density = positions.shape[0] / jnp.prod(jnp.max(positions, axis=0) - jnp.min(positions, axis=0))
        return int(3 * jnp.ceil(approx_density * sphere_volume(self._cutoff)).item())


class PeriodicBoundary(NeighbourFinder):
    _cell: jax.Array
    _cutoff: float
    _cell_list: jax.Array
    _grid_points: jax.Array
    _include_self: bool
    _include_images: bool
    _self_cell: int

    def __init__(
        self,
        cell: jax.typing.ArrayLike,
        cutoff: float,
        max_cell_multiples: int = DEFAULT_MAX_CELL_MULTIPLES,
        include_self=False,
        include_images=True,
    ):
        self._cell = jnp.asarray(cell)
        self._cutoff = cutoff
        self._cell_list, self._grid_points = get_cell_list(self._cell, cutoff, max_cell_multiples=max_cell_multiples)
        self._self_cell = jnp.argwhere(jax.vmap(jnp.array_equal, (0, None))(self._cell_list,
                                                                            jnp.zeros(3, dtype=i32)))[0, 0].item()
        self._include_self = include_self
        self._include_images = include_images

    def get_neighbours(self, positions: jax.typing.ArrayLike, max_neighbours: int = None) -> NeighbourList:
        num_points = positions.shape[0]
        # num_cells = self._cell_list.shape[0]
        max_neighbours = max_neighbours or self.estimate_neighbours(positions)
        print(max_neighbours)

        neighbours = jax.vmap(lambda shift: shift + positions)(self._grid_points).reshape(-1, 3)

        # Get the neighbours mask
        neigh_mask = jax.vmap(neighbours_mask_direct, (0, None, None))(positions, neighbours, self._cutoff)
        # if not self._include_self or not self._include_images:
        #     neigh_mask = neigh_mask.reshape(num_points, num_points, num_cells)
        #     if not self._include_self:
        #         mask = ~jnp.eye(num_points, dtype=bool)
        #         neigh_mask = neigh_mask.at[:, self._self_cell, :].set(neigh_mask[:, self._self_cell, :] & mask)
        #     neigh_mask = neigh_mask.reshape(num_points, num_cells * num_points)

        get_neighbours = functools.partial(jnp.argwhere, size=max_neighbours, fill_value=MASK_VALUE)
        to_idx = jax.vmap(get_neighbours)(neigh_mask)[..., 0]

        # Repeat the cells for each
        cells = jnp.repeat(self._cell_list, num_points, axis=0)
        cell_indices = jax.vmap(jnp.take, (None, 0, None))(cells, to_idx, 0)

        return NeighbourList(
            jnp.where(to_idx == MASK_VALUE, MASK_VALUE, to_idx % num_points),
            cell_indices,
            actual_max_neighbours=jnp.max(neigh_mask.sum(axis=1)),
            finder=self,
        )

    def estimate_neighbours(self, positions: jax.typing.ArrayLike) -> int:
        density = positions.shape[0] / cell_volume(self._cell)
        return int(1.25 * jnp.ceil(density * sphere_volume(self._cutoff)).item())


def generate_positions(cell: jax.Array, positions: jax.Array, cell_shifts: jax.Array) -> jax.Array:
    return jax.vmap(lambda shift: (shift @ cell) + positions)(cell_shifts)


def get_cell_list(cell: jax.typing.ArrayLike,
                  cutoff: float,
                  max_cell_multiples: int = DEFAULT_MAX_CELL_MULTIPLES) -> Tuple[jax.Array, jax.Array]:
    cell = jnp.asarray(cell)

    # Get the multipliers for each cell direction
    cell_ranges = get_cell_multiple_ranges(cell, cutoff=cutoff)
    # TODO: Check cell ranges are within MAX

    cell_grid = jnp.array(
        jnp.meshgrid(
            jnp.arange(*cell_ranges[0]),
            jnp.arange(*cell_ranges[1]),
            jnp.arange(*cell_ranges[2]),
        )
    )
    reshaped = cell_grid.T.reshape(-1, 3)
    grid_points = reshaped @ cell

    # corners = jnp.array(list(itertools.product((0, 1), repeat=3)), dtype=i32)
    # corners = corners @ cell
    # mask = jax.vmap(neighbours_mask_direct, (0, None, None))(corners, grid_points, cutoff).any(axis=0)
    # return reshaped[mask], grid_points[mask]
    return reshaped, grid_points


def get_cell_multiple_range(cell: jax.typing.ArrayLike, cell_vector: int, cutoff: float) -> Tuple[int, int]:
    multiplier = get_max_cell_vector_repetitions(cell, cell_vector, cutoff=cutoff)
    return -math.ceil(multiplier), math.ceil(multiplier) + 1


def get_cell_multiple_ranges(cell: jax.typing.ArrayLike, cutoff: float) -> Tuple[Tuple[int, int]]:
    return tuple(map(functools.partial(get_cell_multiple_range, cell, cutoff=cutoff), (0, 1, 2)))


def get_max_cell_vector_repetitions(cell: jax.typing.ArrayLike, cell_vector: int, cutoff: float) -> float:
    """
    Given a unit cell defined by three vectors this will return the number of multiples of the vector indexed by
    `cell_vector` that are needed to reach the edge of a sphere with radius `cutoff`.  This tells you what multiple
    of cell vectors you need to up to (when rounded up to the nearest integer) in order to fully cover all points
    in the sphere, in teh given cell vector direction.
    """
    cell = jnp.asarray(cell)
    vec1 = (cell_vector + 1) % 3
    vec2 = (cell_vector + 2) % 3
    volume = cell_volume(cell).item()

    vec1_cross_vec2_len = jnp.linalg.norm(jnp.cross(cell[vec1], cell[vec2])).item()
    return get_num_plane_repetitions_to_bound_sphere(cutoff, volume, vec1_cross_vec2_len)


def neighbours_mask_aabb(centre: jax.typing.ArrayLike, neighbours: jax.typing.ArrayLike, cutoff: float) -> jax.Array:
    """Get the indices of all points that are within a cutoff sphere centred on `centre` with a radius `cutoff` using
    the Axis Aligned Bounding Box method"""
    diag = cutoff / jnp.sqrt(3.0)
    centred = neighbours - centre
    # First find those that fit into the axis aligned bounding box that fits within the cutoff sphere
    definitely_neighbour = jnp.array(jnp.all((-diag < centred) & (centred < diag), axis=1), dtype=bool)
    maybe_neighbour = (jnp.all(-cutoff < centred & centred < cutoff, axis=1) & ~definitely_neighbour)

    # Now check the remaining ones that lie within the shell between the AABB that fits within the sphere and the AABB
    # that bounds the sphere
    maybe_norm_sq = jnp.sum(centred[maybe_neighbour]**2, axis=1)
    return definitely_neighbour.at[maybe_neighbour].set(
        definitely_neighbour[maybe_neighbour] | (maybe_norm_sq < (cutoff * cutoff))
    )


def neighbours_mask_direct(centre: jax.typing.ArrayLike, neighbours: jax.typing.ArrayLike, cutoff: float) -> jax.Array:
    """Get the indices of all points that are within a cutoff sphere centred on `centre` with a radius `cutoff` by
    calculating all distance vector norms and masking those within the cutoff"""
    centred = neighbours - centre
    return jnp.array(jnp.sum(centred**2, axis=1) < (cutoff * cutoff), dtype=bool)
