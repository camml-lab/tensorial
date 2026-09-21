"""NumPy-based neighbour-finder backends (open and periodic boundaries)."""

import numbers

from jaxtyping import Bool, Float, Int
import numpy as np
import scipy.spatial.distance as ssd
from typing_extensions import override

from tensorial.typing import Array, CellType, PbcType

from . import distances, unit_cells

__all__ = ("neighbour_finder", "OpenBoundary", "PeriodicBoundary")

Range = tuple[int, int]


class GridCells:
    """A grid of integer image coordinates inside a unit cell."""

    def __init__(
        self,
        cell: Float[Array, "3 3"],
        grid_coords: Int[Array, "... 3"],
        grid_pts: Float[Array, "... 3"] | None = None,
    ):
        """Initialise the grid.

        Args:
            cell: unit cell, rows are the three cell vectors.
            grid_coords: integer image coordinates, shape ``(N, 3)``.
            grid_pts: optional precomputed Cartesian positions of the images.
        """
        self._cell = cell
        self._grid_coords = grid_coords
        self._grid_pts = grid_pts

    @property
    def num_cells(self) -> int:
        """The number of image cells in the grid."""
        return self._grid_coords.shape[0]

    @property
    def cell(self) -> Float[Array, "3 3"]:
        """The unit cell for this grid."""
        return self._cell

    @property
    def grid_coords(self) -> Int[Array, "... 3"]:
        """The integer image coordinates."""
        return self._grid_coords

    @property
    def grid_pts(self) -> Float[Array, "... 3"]:
        """The Cartesian positions of the images, computed lazily from ``grid_coords``."""
        if self._grid_pts is None:
            self._grid_pts = self._grid_coords @ self._cell

        return self._grid_pts

    def mask_off(self, mask: Bool[Array, "... 3"]) -> "GridCells":
        """Return a new :class:`GridCells` with only the entries selected by ``mask``."""
        return GridCells(
            self._cell,
            self._grid_coords[mask],
            self._grid_pts[mask] if self._grid_pts is not None else None,
        )

    def bloom(
        self, points: Float[Array, "N 3"]
    ) -> tuple[Float[Array, "... 3"], Int[Array, "..."], Int[Array, "... 3"]]:
        """Expand a set of points into their images in every cell of the grid.

        Args:
            points: the positions to expand, shape ``(N, 3)``.

        Returns:
            a tuple of (Cartesian image positions, original-point index, cell index).
        """
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
        """Collapse the (possibly multi-dim) grid to a single flat list of cells."""
        self._grid_coords = self._grid_coords.reshape(-1, 3)
        if self._grid_pts is not None:
            self._grid_pts = self._grid_pts.reshape(-1, 3)
        return self


class NeighbourList(distances.NeighbourList):
    """A concrete :class:`~tensorial.geometry.distances.NeighbourList` backed by NumPy."""

    def __init__(self, n_particles: int, edges: distances.Edges):
        """Store the particle count and the edge list."""
        self._n_particles = n_particles
        self._edges = edges

    @override
    @property
    def num_particles(self) -> int:
        """The number of particles in this list."""
        return self._n_particles

    @override
    @property
    def max_neighbours(self) -> int:
        """The maximum degree of any particle in the list."""
        return np.unique(self._edges.from_idx, return_counts=True)[1].max()

    def get_edges(self) -> distances.Edges:
        """The underlying edge list."""
        return self._edges


class OpenBoundary(distances.NeighbourFinder):
    """Open-boundary neighbour finder: a simple cutoff sphere, no periodic images."""

    def __init__(self, cutoff: numbers.Number, include_self: bool = False):
        """Initialise with a cutoff radius and (optionally) include self-neighbours."""
        self._cutoff = float(cutoff)
        self._include_self = include_self

    def get_neighbours(
        self,
        positions: Float[Array, "N 3"],
        max_neighbours: int = None,  # pylint: disable=unused-argument
    ) -> distances.NeighbourList:
        """All (i, j) pairs with ``|p_i - p_j| < cutoff`` (and self-neighbours if requested)."""
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
    """Periodic-boundary neighbour finder, considering images in periodic cell repetitions."""

    def __init__(
        self,
        cell: CellType,
        cutoff: numbers.Number,
        pbc: PbcType,
        *,
        include_self=False,
        include_images=True,
    ):
        """
        Args:
            cell: the unit cell
            cutoff: the cutoff radius that defines if an atom is a
                neighbour or not
            pbc: specifies which unit cell vectors are to be considered
                periodic
            include_self: include an atom as its own neighbour within
                the central unit cell
            include_images: include images of an atom in periodic
                repetitions of the central unit cell as neighbours
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
        self,
        positions: Float[Array, "N 3"],
        max_neighbours: int = None,  # pylint: disable=unused-argument
    ) -> distances.NeighbourList:
        """Return the neighbour list across periodic images within the cutoff."""
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


def neighbour_finder(
    cutoff: numbers.Number,
    cell: CellType | None = None,
    pbc: PbcType | None = None,
    include_self: bool = False,
    **kwargs,
) -> distances.NeighbourFinder:
    """Create the appropriate neighbour finder for the given boundary conditions.

    Pick :class:`PeriodicBoundary` if any component of ``pbc`` is ``True`` (and a
    cell was provided), otherwise :class:`OpenBoundary`.

    Args:
        cutoff: the cutoff radius.
        cell: the unit cell required for periodic boundaries.
        pbc: which cell directions are periodic.
        include_self: also count self-neighbours (same particle, central cell).
        **kwargs: forwarded to the chosen backend.

    Returns:
        a :class:`~tensorial.geometry.distances.NeighbourFinder` instance.
    """
    if pbc is not None and any(pbc):
        return PeriodicBoundary(cell, cutoff, pbc, include_self=include_self, **kwargs)

    return OpenBoundary(cutoff, include_self=include_self)
