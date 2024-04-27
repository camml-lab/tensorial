# -*- coding: utf-8 -*-
import ase.build
import jax
import jax.numpy as jnp
import jax.random
import numpy as np
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


def test_species_transform(h2coh: ase.Atoms):  # pylint: disable=redefined-outer-name
    atomic_numbers = tuple(set(h2coh.numbers))
    num_atoms = len(h2coh)
    graph = atomic.graph_from_ase(h2coh, r_max=3.0)

    transform = atomic.SpeciesTransform(atomic_numbers)
    # params = transform.init(rng_key, graph)
    # out = transform.apply(params, graph)
    out = transform(graph)

    transformed = jnp.array(list(map(
        atomic_numbers.index,
        h2coh.numbers,
    )))
    assert out.nodes[transform.out_field].shape == (num_atoms,)
    assert jnp.all(out.nodes[transform.out_field] == transformed)


def test_per_species_rescale():
    molecule = ase.build.molecule('SiH4')
    types = list(set(molecule.get_atomic_numbers()))
    energies = np.random.rand(len(molecule))
    molecule.arrays[atomic.ENERGY_PER_ATOM] = energies

    molecule_graph = atomic.graph_from_ase(molecule, r_max=2.0, atom_include_keys=('numbers', atomic.ENERGY_PER_ATOM))
    # Update the graph with the type indexes
    species_transform = atomic.SpeciesTransform(types)
    molecule_graph = species_transform(molecule_graph)

    rescale = atomic.per_species_rescale(
        len(types),
        field=f'nodes.{atomic.ENERGY_PER_ATOM}',
    )
    params = rescale.init(jax.random.key(0), molecule_graph)
    rescaled = rescale.apply(params, molecule_graph)

    # Check before and after
    assert jnp.all(molecule_graph.nodes[atomic.ENERGY_PER_ATOM] == energies)
    assert jnp.all(rescaled.nodes[atomic.ENERGY_PER_ATOM] != energies)
