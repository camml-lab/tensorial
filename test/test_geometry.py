import math

import ase.cell
import ase.neighborlist
import equinox
import jax.numpy as jnp
import numpy as np
import pytest

from tensorial import geometry
from tensorial.geometry import jax_neighbours


@pytest.mark.parametrize("self_interaction", [True, False])
def test_jax_open_boundary(self_interaction):
    n_points = 100
    cutoff = 0.2
    positions = np.random.rand(n_points, 3)

    neighbours = []
    for i, r_i in enumerate(positions):
        r_j = positions - r_i
        norms_sq = np.sum(r_j**2, axis=1)
        mask = norms_sq < (cutoff * cutoff)
        if not self_interaction:
            mask[i] = False
        neighbours.append(np.argwhere(mask)[:, 0])
        neighbours[-1].sort()

    free = geometry.jax_neighbours.OpenBoundary(cutoff, include_self=self_interaction)
    get_neighbours = equinox.filter_jit(free.get_neighbours)
    nlist = get_neighbours(positions, max_neighbours=free.estimate_neighbours(positions))
    assert (
        not nlist.did_overflow
    ), f"Neighbour list has overflown, need at least {nlist.actual_max_neighbours} max neighbours"

    from_idx, to_idx, cells = nlist.get_edges()
    # This has open boundary conditions so cells should all be zero
    assert np.all(cells == np.zeros((len(from_idx), 3)))
    for i, i_neighbours in enumerate(neighbours):
        from_neighbour_list = to_idx[from_idx == i]
        from_neighbour_list.sort()
        assert np.all(from_neighbour_list == i_neighbours)


@pytest.mark.parametrize("cell_angles", [(90.0, 90.0), (60.0, 135.0)])
def test_get_cell_list(cell_angles):
    cutoff = 1.5
    ase_cell = ase.cell.Cell.new([1, 1.0, 1.0, *np.random.uniform(*cell_angles, size=3)])
    cell = ase_cell.array

    cell_list = jax_neighbours.get_cell_list(cell, cutoff=cutoff)
    cell_list = tuple(
        map(np.array, cell_list)
    )  # Convert to numpy to make things easier to work with

    # Check that cell list has not returned any duplicate points in the grid
    assert len(np.unique(cell_list[0], axis=0)) == len(cell_list[0])

    cell_ranges = geometry.unit_cells.get_cell_multiple_ranges(cell, cutoff=cutoff)

    grid_idxs = []
    grid_pts = []
    for i in range(*cell_ranges[0]):
        for j in range(*cell_ranges[1]):
            for k in range(*cell_ranges[2]):
                grid_idx = np.array([i, j, k])
                grid_pt = i * cell[0] + j * cell[1] + k * cell[2]

                grid_idxs.append(grid_idx)
                grid_pts.append(grid_pt)

    grid_idxs = np.vstack(grid_idxs)
    grid_pts = np.vstack(grid_pts)

    sort_indices = np.argsort(grid_idxs.view("i8,i8,i8"), order=("f0", "f1", "f2"), axis=0)
    cell_list_sort_indices = np.argsort(
        cell_list[0].view("i8,i8,i8"), order=("f0", "f1", "f2"), axis=0
    )

    assert np.all(cell_list[0][cell_list_sort_indices] == grid_idxs[sort_indices])
    assert np.allclose(cell_list[1][cell_list_sort_indices], grid_pts[sort_indices])


@pytest.mark.parametrize("cell_angles", [(90.0, 90.0), (60.0, 135.0)])
def test_ase_neighbour_list(cell_angles):
    n_points = 20
    cutoff = 1.5
    cutoff_sq = cutoff**2
    ase_cell = ase.cell.Cell.new([1, 1.0, 1.0, *np.random.uniform(*cell_angles, size=3)])
    pts_frac = np.random.rand(n_points, 3)
    cell = ase_cell.array
    positions = pts_frac @ cell

    cell_list = jax_neighbours.get_cell_list(cell, cutoff=cutoff)
    position_copies = []
    # for grid_point in cell_list[1]:
    #     position_copies.append(positions + grid_point)
    for grid_idx in cell_list[0]:
        grid_point = grid_idx @ cell
        position_copies.append(positions + grid_point)
    position_copies = np.vstack(position_copies)

    # Find neighbours from all points in original cell to all copies (including self)
    neighbours = []
    for position in positions:
        diffs = position_copies - position
        neighbours.append(position_copies[np.sum(diffs**2, axis=1) < cutoff_sq])
    neighbours = np.vstack(neighbours)

    ase_edges = geometry.Edges(
        *ase.neighborlist.primitive_neighbor_list(
            "ijS",
            pbc=[True, True, True],
            cell=cell,
            positions=positions,
            cutoff=cutoff,
            self_interaction=True,
        )
    )
    assert len(ase_edges.to_idx) == len(neighbours)


@pytest.mark.parametrize("self_interaction", [True, False])
@pytest.mark.parametrize("cutoff,cell_angles", [(1.5, (90.0, 90.0)), (1.5, (60.0, 125.0))])
@pytest.mark.parametrize("pbc", [(True, True, True), (True, False, True), (True, False, False)])
def test_jax_periodic_boundary(self_interaction, cutoff, cell_angles, pbc):
    n_points = 12
    cell_ = ase.cell.Cell.new([1, 1.0, 1.0, *np.random.uniform(*cell_angles, size=3)])
    pts_frac = np.random.rand(n_points, 3)
    cell = cell_.array
    positions = pts_frac @ cell

    periodic = jax_neighbours.PeriodicBoundary(cell, cutoff, pbc=pbc, include_self=self_interaction)
    get_neighbours = equinox.filter_jit(periodic.get_neighbours)
    # get_neighbours = periodic.get_neighbours
    neighbours = get_neighbours(positions, max_neighbours=periodic.estimate_neighbours(positions))
    assert not neighbours.did_overflow, (
        f"Neighbour list has overflown, need at least {neighbours.actual_max_neighbours} max "
        f"neighbours"
    )

    tensorial_edges = neighbours.get_edges()
    edge_vecs = geometry.unit_cells.get_edge_vectors(positions, tensorial_edges, cell)
    assert np.all(
        np.sum(edge_vecs**2, axis=1) <= (cutoff * cutoff)
    ), "Edges returned that are longer than the cutoff"

    # Compare to results from ASE
    ase_edges = geometry.Edges(
        *ase.neighborlist.primitive_neighbor_list(
            "ijS",
            pbc=pbc,
            cell=cell,
            positions=positions,
            cutoff=cutoff,
            self_interaction=self_interaction,
        )
    )

    assert len(tensorial_edges.from_idx) == len(
        ase_edges.from_idx
    ), "Number of edges don't match result from ASE"
    for i in range(len(positions)):
        tensorial_neighs = tensorial_edges.to_idx[tensorial_edges.from_idx == i]
        ase_neighs = ase_edges.to_idx[ase_edges.from_idx == i]
        assert len(tensorial_neighs) == len(ase_neighs)
        assert np.all(np.sort(tensorial_neighs) == np.sort(ase_neighs))


@pytest.mark.parametrize("self_interaction", [True, False])
def test_np_open_boundary(self_interaction):
    n_points = 100
    cutoff = 0.2
    positions = np.random.rand(n_points, 3)

    neighbours = []
    for i, r_i in enumerate(positions):
        r_j = positions - r_i
        norms_sq = np.sum(r_j**2, axis=1)
        mask = norms_sq < (cutoff * cutoff)
        if not self_interaction:
            mask[i] = False
        neighbours.append(np.argwhere(mask)[:, 0])
        neighbours[-1].sort()

    free = geometry.np_neighbours.OpenBoundary(cutoff, include_self=self_interaction)
    nlist = free.get_neighbours(positions)

    from_idx, to_idx, cells = nlist.get_edges()
    # This has open boundary conditions so cells should all be zero
    assert np.all(cells == np.zeros((len(from_idx), 3)))
    for i, i_neighbours in enumerate(neighbours):
        from_neighbour_list = to_idx[from_idx == i]
        from_neighbour_list.sort()
        assert np.all(from_neighbour_list == i_neighbours)


@pytest.mark.parametrize("self_interaction", [True, False])
@pytest.mark.parametrize("cutoff,cell_angles", [(1.5, (90.0, 90.0)), (1.5, (60.0, 125.0))])
@pytest.mark.parametrize("pbc", [(True, True, True), (True, False, True), (True, False, False)])
def test_np_periodic_boundary(self_interaction, cutoff, cell_angles, pbc):
    n_points = 12
    cell_ = ase.cell.Cell.new([1, 1.0, 1.0, *np.random.uniform(*cell_angles, size=3)])
    pts_frac = np.random.rand(n_points, 3)
    cell = cell_.array
    positions = pts_frac @ cell

    periodic = geometry.np_neighbours.PeriodicBoundary(
        cell, cutoff, pbc=pbc, include_images=True, include_self=self_interaction
    )
    neighbours = periodic.get_neighbours(positions)

    tensorial_edges = neighbours.get_edges()
    edge_vecs = geometry.unit_cells.get_edge_vectors(positions, tensorial_edges, cell)
    assert np.all(
        np.sum(edge_vecs**2, axis=1) <= (cutoff * cutoff)
    ), "Edges returned that are longer than the cutoff"

    # Compare to results from ASE
    ase_edges = geometry.Edges(
        *ase.neighborlist.primitive_neighbor_list(
            "ijS",
            pbc=pbc,
            cell=cell,
            positions=positions,
            cutoff=cutoff,
            self_interaction=self_interaction,
        )
    )

    assert len(tensorial_edges.from_idx) == len(
        ase_edges.from_idx
    ), "Number of edges don't match result from ASE"
    for i in range(len(positions)):
        tensorial_neighs = tensorial_edges.to_idx[tensorial_edges.from_idx == i]
        ase_neighs = ase_edges.to_idx[ase_edges.from_idx == i]
        assert len(tensorial_neighs) == len(ase_neighs)
        assert np.all(np.sort(tensorial_neighs) == np.sort(ase_neighs))


def test_neighbour_list_reallocate():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    nlist = jax_neighbours.OpenBoundary(cutoff=2.0, include_self=True).get_neighbours(
        positions, max_neighbours=1
    )
    assert nlist.did_overflow
    assert nlist.max_neighbours == 1

    # Try reallocating to see if we can get the correct number of neighbours
    nlist2 = nlist.reallocate(positions)
    assert not nlist2.did_overflow


def test_get_max_cell_vector_repetitions():
    abc = [1.0, 2.0, 3.0]
    cutoff = 0.9
    cell = np.array([[abc[0], 0.0, 0.0], [0.0, abc[1], 0.0], [0.0, 0.0, abc[2]]])

    for cell_vector in range(3):
        repetitions = geometry.unit_cells.get_max_cell_vector_repetitions(
            cell, cell_vector, cutoff=cutoff
        )
        assert np.isclose(repetitions, cutoff / abc[cell_vector])


@pytest.mark.parametrize("cutoff", (0.0, 1.5))
def test_get_cell_multipliers(cutoff):
    abc = [1.0, 2.0, 3.0]
    cell = np.array([[abc[0], 0.0, 0.0], [0.0, abc[1], 0.0], [0.0, 0.0, abc[2]]])

    for cell_vector in range(3):
        mul_range = geometry.unit_cells.get_cell_multiple_range(cell, cell_vector, cutoff=cutoff)
        expected = (
            -math.ceil(cutoff / abc[cell_vector]),
            math.ceil(cutoff / abc[cell_vector]) + 1,
        )
        assert jnp.all(mul_range == expected)
