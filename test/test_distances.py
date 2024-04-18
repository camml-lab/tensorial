# -*- coding: utf-8 -*-
import ase.cell
import ase.neighborlist
import equinox
import numpy as np

from tensorial import distances
import tensorial.distances


def test_open_boundary():
    n_points = 100
    cutoff = 0.2
    positions = np.random.rand(n_points, 3)

    neighbours = []
    for i, r_i in enumerate(positions):
        r_j = positions - r_i
        norms_sq = np.sum(r_j**2, axis=1)
        neighbours.append(np.argwhere(norms_sq < (cutoff * cutoff))[:, 0])
        neighbours[-1].sort()

    free = distances.OpenBoundary(cutoff)
    get_neighbours = equinox.filter_jit(free.get_neighbours)
    nlist = get_neighbours(positions, max_neighbours=free.estimate_neighbours(positions))
    assert not nlist.did_overflow, f'Neighbour list has overflown, need at least {nlist.actual_max_neighbours} max neighbours'

    from_idx, to_idx, cells = nlist.get_edges(self_interaction=True)
    # This has open boundary conditions so cells should all be zero
    assert np.all(cells == np.zeros((len(from_idx), 3)))
    for i, i_neighbours in enumerate(neighbours):
        from_neighbour_list = to_idx[from_idx == i]
        from_neighbour_list.sort()
        assert np.all(from_neighbour_list == i_neighbours)


def test_periodic_boundary():
    n_points = 30
    cutoff = 1.05
    cell_ = ase.cell.Cell.new([1, 1., 1., *np.random.uniform(50., 145, size=3)])
    pts_frac = np.random.rand(n_points, 3)
    cell = cell_.array
    positions = pts_frac @ cell

    periodic = distances.PeriodicBoundary(cell, cutoff)
    # get_neighbours = equinox.filter_jit(periodic.get_neighbours)
    get_neighbours = periodic.get_neighbours
    neighbours = get_neighbours(positions, max_neighbours=periodic.estimate_neighbours(positions))
    assert not neighbours.did_overflow, f'Neighbour list has overflown, need at least {neighbours.actual_max_neighbours} max neighbours'

    tensorial_edges = neighbours.get_edges(self_interaction=True)
    edge_vecs = tensorial.distances.get_edge_vectors(positions, tensorial_edges, cell)
    assert np.all(np.sum(edge_vecs**2, axis=1) < (cutoff * cutoff)), 'Edges returned that are longer than the cutoff'

    # Compare to results from ASE
    ase_edges = tensorial.distances.Edges(
        *ase.neighborlist.primitive_neighbor_list(
            'ijS', pbc=[True, True, True], cell=cell, positions=positions, cutoff=cutoff, self_interaction=False
        )
    )

    assert np.all(ase_edges.from_idx == tensorial_edges.from_idx), "Number of edges don't match result from ASE"
    for i in range(len(positions)):
        tensorial_neighs = tensorial_edges.to_idx[tensorial_edges.from_idx == i]
        ase_neighs = ase_edges.to_idx[ase_edges.from_idx == i]
        assert np.all(np.sort(tensorial_neighs) == np.sort(ase_neighs))


def test_neighbour_list_reallocate():
    positions = np.array([[0., 0., 0.], [1., 1., 1.]])
    nlist = distances.OpenBoundary(cutoff=2.).get_neighbours(positions, max_neighbours=1)
    assert nlist.did_overflow
    assert nlist.max_neighbours == 1

    # Try reallocating to see if we can get the correct number of neighbours
    nlist2 = nlist.reallocate(positions)
    assert not nlist2.did_overflow
