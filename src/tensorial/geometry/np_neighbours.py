import numbers
from typing import Optional

import jax.typing
import jaxtyping as jt
import numpy as np
import scipy.spatial.distance as ssd
from typing_extensions import override

from . import distances, unit_cells
from .. import typing

__all__ = ("neighbour_finder", "OpenBoundary", "PeriodicBoundary")

Range = tuple[int, int]


class GridCells:
    def __init__(
        self,
        cell: jt.Float[jt.Array, "3 3"],
        grid_coords: jt.Int[jt.ArrayLike, "... 3"],
        grid_pts: Optional[jt.Float[jt.ArrayLike, "... 3"]] = None,
    ):
        self._cell = cell
        self._grid_coords = grid_coords
        self._grid_pts = grid_pts

    @property
    def num_cells(self) -> int:
        return self._grid_coords.shape[0]

    @property
    def cell(self) -> jt.Float[jt.Array, "3 3"]:
        return self._cell

    @property
    def grid_coords(self) -> jt.Int[jt.Array, "... 3"]:
        return self._grid_coords

    @property
    def grid_pts(self) -> jt.Float[jt.Array, "... 3"]:
        if self._grid_pts is None:
            self._grid_pts = self._grid_coords @ self._cell

        return self._grid_pts

    def mask_off(self, mask: jt.Bool[jt.Array, "... 3"]) -> "GridCells":
        return GridCells(
            self._cell,
            self._grid_coords[mask],
            self._grid_pts[mask] if self._grid_pts is not None else None,
        )

    def bloom(
        self, points: jt.Float[jt.ArrayLike, "N 3"]
    ) -> tuple[
        jt.Float[jt.ArrayLike, "... 3"], jt.Int[jt.ArrayLike, "..."], jt.Int[jt.ArrayLike, "... 3"]
    ]:
        bloomed_points = []
        for grid_pt in self.grid_pts:
            bloomed_points.extend(grid_pt + points)
        bloomed_points = np.array(bloomed_points)

        n_pts = points.shape[0]
        return (
            bloomed_points,
            np.broadcast_to(
                np.arange(points.shape[0]), (self.grid_coords.shape[0], n_pts)
            ).flatten(),
            # np.repeat(np.arange(points.shape[0], dtype=int), self.grid_coords.shape[0]),
            np.repeat(self.grid_coords, n_pts, axis=0),
        )

    def flatten(self):
        self._grid_coords = self._grid_coords.reshape(-1, 3)
        if self._grid_pts is not None:
            self._grid_pts = self._grid_pts.reshape(-1, 3)
        return self


class NeighbourList(distances.NeighbourList):
    def __init__(self, n_particles: int, edges: distances.Edges):
        self._n_particles = n_particles
        self._edges = edges

    @override
    @property
    def num_particles(self) -> int:
        return self._n_particles

    @override
    @property
    def max_neighbours(self) -> int:
        return np.unique(self._edges.from_idx, return_counts=True)[1].max()

    def get_edges(self) -> distances.Edges:
        return self._edges


class OpenBoundary(distances.NeighbourFinder):
    def __init__(self, cutoff: numbers.Number, include_self: bool = False):
        self._cutoff = float(cutoff)
        self._include_self = include_self

    def get_neighbours(
        self, positions: jt.Float[jax.typing.ArrayLike, "N 3"], max_neighbours: int = None
    ) -> distances.NeighbourList:
        npts = positions.shape[0]
        dists = ssd.squareform(ssd.pdist(positions))
        mask = dists < self._cutoff
        if not self._include_self:
            mask = mask & ~np.eye(npts, dtype=bool)

        res = np.argwhere(mask)
        return NeighbourList(
            npts, distances.Edges(res[:, 0], res[:, 1], np.zeros((res.shape[0], 3)))
        )


class PeriodicBoundary(distances.NeighbourFinder):
    def __init__(
        self,
        cell: typing.CellType,
        cutoff: numbers.Number,
        pbc: typing.PbcType,
        *,
        include_self=False,
        include_images=True,
    ):
        """
        :param cell: the unit cell
        :param cutoff: the cutoff radius that defines if an atom is a neighbour or not
        :param pbc: specifies which unit cell vectors are to be considered periodic
        :param include_self: include an atom as its own neighbour within the central unit cell
        :param include_images: include images of an atom in periodic repetitions of the central
            unit cell as neighbours
        """
        self._cell = cell
        self._cutoff = cutoff
        self._pbc = pbc
        self._include_self = include_self
        self._include_images = include_images

        # Create the integer grid of points representing the lattice
        ranges: tuple[Range, Range, Range] = unit_cells.get_cell_multiple_ranges(cell, cutoff, pbc)
        aspace, bspace, cspace = tuple(
            map(
                lambda x0x1: np.linspace(*x0x1, x0x1[1] - x0x1[0], endpoint=False, dtype=int),
                ranges,
            )
        )
        av, bv, cv = np.meshgrid(aspace, bspace, cspace)
        grid_idx = np.stack((av, bv, cv), axis=-1)
        self._full_grid = GridCells(cell, grid_idx)
        self._full_grid.flatten()

    def get_neighbours(
        self, positions: jt.Float[jax.typing.ArrayLike, "N 3"], max_neighbours: int = None
    ) -> distances.NeighbourList:
        n_pts: int = positions.shape[0]

        neigh_pos, neigh_idx, neigh_grid_coords = self._full_grid.bloom(positions)
        valid_masks = ssd.cdist(positions, neigh_pos) < self._cutoff

        from_idx = []
        to_idx = []
        cell_idx = []
        for i, mask in enumerate(valid_masks):
            if not self._include_images:
                mask &= ~((neigh_idx == i) & ~np.all(neigh_grid_coords == [0, 0, 0], axis=1))
            if not self._include_self:
                mask &= ~((neigh_idx == i) & np.all(neigh_grid_coords == [0, 0, 0], axis=1))

            n_neighbours = np.sum(mask).item()
            from_idx.append(np.full(n_neighbours, i, dtype=int))
            to_idx.append(neigh_idx[mask])
            cell_idx.append(neigh_grid_coords[mask])

        return NeighbourList(
            n_pts, distances.Edges(np.hstack(from_idx), np.hstack(to_idx), np.vstack(cell_idx))
        )


# too slow: @jt.jaxtyped(typechecker=beartype.beartype)
def neighbour_finder(
    cutoff: numbers.Number,
    cell: Optional[typing.CellType] = None,
    pbc: Optional[typing.PbcType] = None,
    include_self: bool = False,
    **kwargs,
) -> distances.NeighbourFinder:
    if pbc is not None and any(pbc):
        return PeriodicBoundary(cell, cutoff, pbc, include_self=include_self, **kwargs)

    return OpenBoundary(cutoff, include_self=include_self)
