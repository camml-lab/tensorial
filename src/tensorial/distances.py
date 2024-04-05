# -*- coding: utf-8 -*-
import math
from typing import Callable, Tuple

import equinox
import jax
import jax.numpy as jnp

DistancesFunction = Callable[[jax.Array, jax.Array], Tuple[jax.Array, jax.Array]]


def get_num_plane_repetitions_to_bound_sphere(radius: float, volume: float, cross_len: float) -> float:
    # The vector normal to the plane
    return radius / volume * cross_len


def free(cutoff: float = None) -> DistancesFunction:
    """Return a distances function for free space (with an optional cutoff)"""
    cutoff_sq = jnp.inf if cutoff is None else cutoff * cutoff

    def calc_dr(r1: jax.Array, r2: jax.Array):
        dr = r2 - r1
        if jnp.dot(dr, dr) > cutoff_sq:
            return jnp.zeros((0, 3)), jnp.zeros((0, 3), dtype=int)

        return dr.reshape(1, 3), jnp.zeros((1, 3), dtype=int)

    return calc_dr


def periodic(cell: jax.Array, cutoff: float) -> DistancesFunction:
    """Return a distances function for a periodic space defined by a unit cell

    :param cell: a `[3, 3]` array of unti cell vectors in row-major format
    :param cutoff: a cutoff distance for the maximum distance considered
    :return: the distances function
    """
    return Periodic(cell, cutoff=cutoff)


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


def get_neighbour_list(
    distances_fn: DistancesFunction,
    positions: jax.Array,
    self_interaction=False,
    strict_self_interaction=True,
):
    from_idx = []
    to_idx = []
    edge_vectors = []
    cell_shifts = []

    for i, r1 in enumerate(positions):
        for j, r2 in enumerate(positions[:i]):
            drs, cell_indices = distances_fn(r1, r2)
            n_vecs = len(drs)
            from_idx.extend([i] * n_vecs)
            to_idx.extend([j] * n_vecs)
            edge_vectors.extend(drs)
            cell_shifts.extend(cell_indices)

            # Now do the symmetric 'back' edges, from j to i
            from_idx.extend([j] * n_vecs)
            to_idx.extend([i] * n_vecs)
            edge_vectors.extend(-drs)
            cell_shifts.extend(-cell_indices)

        if strict_self_interaction:
            drs, cell_indices = distances_fn(r1, r1)
            if not self_interaction:
                # Find the ones that are not in the [0, 0, 0] cell
                mask = ~jax.vmap(jnp.array_equal, (0, None))(cell_indices, jnp.array([0, 0, 0]))
                drs = drs[mask]
                cell_indices = cell_indices[mask]

            n_vecs = len(drs)
            from_idx.extend([i] * n_vecs)
            to_idx.extend([i] * n_vecs)
            edge_vectors.extend(drs)
            cell_shifts.extend(cell_indices)

    if not from_idx:
        return (
            jnp.zeros(0, dtype=int),
            jnp.zeros(0, dtype=int),
            jnp.zeros((0, 3)),
            jnp.zeros((0, 3), dtype=int),
        )

    # Turn everything into tensors
    return (
        jnp.array(from_idx),
        jnp.array(to_idx),
        jnp.vstack(edge_vectors),
        jnp.vstack(cell_shifts),
    )
