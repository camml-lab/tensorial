# -*- coding: utf-8 -*-
import ase.build
import jax.numpy as jnp
import pytest

from tensorial.gcnn import atomic


@pytest.fixture
def ase_cubic_si() -> ase.Atoms:
    return ase.build.bulk('Si', 'sc', a=1.0)


@pytest.fixture
def h2coh() -> ase.Atoms:
    return ase.build.molecule('H2COH')


def test_graph_from_ase(ase_cubic_si):  # pylint: disable=redefined-outer-name
    si_graph = atomic.graph_from_ase(ase_cubic_si, r_max=1.1)

    # Graph
    assert si_graph.n_node == jnp.array([1])
    assert si_graph.n_edge == jnp.array([6])
    assert si_graph.senders.shape == (6,)
    assert si_graph.receivers.shape == (6,)

    # Globals
    assert jnp.all(si_graph.globals[atomic.PBC] == jnp.array([[True, True, True]]))

    # Nodes
    assert si_graph.nodes[atomic.ATOMIC_NUMBERS] == jnp.array([14.0])


def test_species_transform(h2coh: ase.Atoms, rng_key):  # pylint: disable=redefined-outer-name
    atomic_numbers = tuple(set(h2coh.numbers))
    num_atoms = len(h2coh)
    graph = atomic.graph_from_ase(h2coh, r_max=3.0)

    transform = atomic.SpeciesTransform(atomic_numbers)
    params = transform.init(rng_key, graph)
    out = transform.apply(params, graph)

    transformed = jnp.array(list(map(
        atomic_numbers.index,
        h2coh.numbers,
    )))
    assert out.nodes[transform.out_field].shape == (num_atoms,)
    assert jnp.all(out.nodes[transform.out_field] == transformed)
