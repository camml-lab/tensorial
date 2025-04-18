from collections.abc import Sequence

import ase.build
import jax
import jax.numpy as jnp
import jax.random
import jraph
import numpy as np
import pytest
import reax

import tensorial
from tensorial import gcnn
from tensorial.gcnn import atomic


@pytest.fixture
def ase_cubic_si() -> "ase.Atoms":
    return ase.build.bulk("Si", "sc", a=1.0)


@pytest.fixture
def h2coh() -> "ase.Atoms":
    return ase.build.molecule("H2COH")


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


def test_graph_from_pymatgen(ase_cubic_si):  # pylint: disable=redefined-outer-name
    pymatgen = pytest.importorskip("pymatgen")
    import pymatgen.io.ase

    si_structure: "pymatgen.Structure" = pymatgen.io.ase.AseAtomsAdaptor.get_structure(ase_cubic_si)
    si_graph = atomic.graph_from_pymatgen(si_structure, r_max=1.1)

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
    atomic_numbers = np.unique(h2coh.numbers).tolist()
    num_atoms = len(h2coh)
    graph = atomic.graph_from_ase(h2coh, r_max=3.0)

    transform = atomic.SpeciesTransform(atomic_numbers)
    # params = transform.init(rng_key, graph)
    # out = transform.apply(params, graph)
    out = transform(graph)

    transformed = jnp.array(
        list(
            map(
                atomic_numbers.index,
                h2coh.numbers,
            )
        )
    )
    assert out.nodes[transform.out_field].shape == (num_atoms,)
    assert jnp.all(out.nodes[transform.out_field] == transformed)


def test_per_species_rescale():
    molecule = ase.build.molecule("SiH4")
    types = np.unique(molecule.get_atomic_numbers())
    energies = np.random.rand(len(molecule))
    molecule.arrays[atomic.keys.ENERGY_PER_ATOM] = energies

    molecule_graph = atomic.graph_from_ase(
        molecule, r_max=2.0, atom_include_keys=("numbers", atomic.keys.ENERGY_PER_ATOM)
    )
    # Update the graph with the type indexes
    species_transform = atomic.SpeciesTransform(types)
    molecule_graph = species_transform(molecule_graph)

    rescale = atomic.per_species_rescale(
        len(types),
        field=f"nodes.{atomic.keys.ENERGY_PER_ATOM}",
    )
    params = rescale.init(jax.random.key(0), molecule_graph)
    rescaled = rescale.apply(params, molecule_graph)

    # Check before and after
    assert jnp.all(molecule_graph.nodes[atomic.keys.ENERGY_PER_ATOM] == energies)
    assert jnp.all(rescaled.nodes[atomic.keys.ENERGY_PER_ATOM] != energies)


def test_metrics(molecule_dataset: Sequence[jraph.GraphsTuple]):
    batch_size = 4
    all_molecules = jraph.batch(molecule_dataset)
    batcher = gcnn.data.GraphLoader(molecule_dataset, batch_size=batch_size)

    avg_num_neighbours = jnp.mean(jnp.bincount(all_molecules.senders))
    assert jnp.allclose(
        avg_num_neighbours,
        reax.metrics.get("atomic/avg_num_neighbours").create(all_molecules).compute(),
    )

    metrics = [
        "atomic/num_species",
        "atomic/all_atomic_numbers",
        "atomic/avg_num_neighbours",
        "atomic/force_std",
    ]

    for name in metrics:
        metric = reax.metrics.get(name)
        # Compute using the data loader
        res = tensorial.metrics.Evaluator(metric).evaluate(batcher)
        # Compute directly
        value = metric.create(all_molecules).compute()

        assert jnp.allclose(res, value), f"Problem with metric {name}"
