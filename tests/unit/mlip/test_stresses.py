from typing import Any, cast

import ase
import jax
import jax.numpy as jnp
import jraph
import numpy as np

from tensorial import gcnn, mlip
from tensorial.gcnn import atomic


def test_calc_stresses_skips_graphs_without_cell():
    """Molecules have no unit cell, so there is no stress to calculate."""
    graph = jraph.GraphsTuple(
        nodes={"positions": jnp.zeros((2, 3))},
        edges={},
        receivers=jnp.array([0]),
        senders=jnp.array([1]),
        globals={},
        n_node=jnp.array([2]),
        n_edge=jnp.array([1]),
    )

    def energy_fn(g: jraph.GraphsTuple) -> jraph.GraphsTuple:
        return g._replace(globals={"predicted_energy": jnp.array([1.0])})

    out = cast(jraph.GraphsTuple, mlip.CalcStresses(energy_fn=energy_fn).apply({}, graph))
    # The energy is still evaluated, but there are no virials or stresses
    assert set(cast(dict, out.globals)) == {"predicted_energy"}


def test_calc_stresses_matches_finite_difference_for_unwrapped_positions():
    """Regression test: the virial is dE/d(strain) with the cell strained along with the
    positions.  Straining only the positions dropped the periodic-image part of the edges that
    cross the cell boundary, giving a wrong (and asymmetric) virial that also changed when atoms
    were moved out of the cell.  Both graphs below are the same structure, one with an atom
    shifted out of the cell, so both must give the finite-difference virial."""
    r_max = 3.0

    def energy_fn(g: jraph.GraphsTuple) -> jraph.GraphsTuple:
        # A pair energy that goes smoothly to zero at the cutoff, so finite differences are
        # not spoiled by edges crossing it
        g = gcnn.with_edge_vectors(g, as_irreps_array=False)
        lengths = cast(dict[str, Any], g.edges)[gcnn.keys.EDGE_LENGTHS][:, 0]
        pair = jnp.where(lengths < r_max, (r_max - lengths) ** 3, 0.0)
        energy = jraph.segment_sum(
            pair,
            jnp.repeat(jnp.arange(len(g.n_edge)), g.n_edge, total_repeat_length=len(pair)),
            num_segments=len(g.n_edge),
        )
        globals_dict = dict(cast(dict, g.globals))
        globals_dict[mlip.keys.predicted(mlip.keys.TOTAL_ENERGY)] = energy[:, None]
        return g._replace(globals=globals_dict)

    def to_graph(atoms: ase.Atoms) -> jraph.GraphsTuple:
        return atomic.graph_from_ase(atoms, r_max, use_calculator=False)

    def energy(atoms: ase.Atoms) -> float:
        graph = jax.tree_util.tree_map(jnp.asarray, to_graph(atoms))
        globals_dict = cast(dict[str, Any], energy_fn(graph).globals)
        return float(globals_dict[mlip.keys.predicted(mlip.keys.TOTAL_ENERGY)][0, 0])

    rng = np.random.default_rng(1)
    cell = np.array([[3.0, 0.0, 0.0], [1.0, 3.2, 0.0], [0.5, 0.4, 3.1]])
    frac = rng.uniform(0.0, 1.0, (4, 3))
    unwrapped = frac.copy()
    unwrapped[0] += [2, -1, 1]
    structures = [
        ase.Atoms("H4", scaled_positions=f, cell=cell, pbc=True) for f in (frac, unwrapped)
    ]

    # Finite-difference dE/d(strain), straining the cell and scaling the atoms with it
    h = 1e-5
    expected = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            eps = np.zeros((3, 3))
            eps[a, b] = h
            plus, minus = structures[0].copy(), structures[0].copy()
            plus.set_cell(cell @ (np.eye(3) + eps), scale_atoms=True)
            minus.set_cell(cell @ (np.eye(3) - eps), scale_atoms=True)
            expected[a, b] = (energy(plus) - energy(minus)) / (2 * h)

    batch = jax.tree_util.tree_map(jnp.asarray, jraph.batch([to_graph(s) for s in structures]))
    out = cast(jraph.GraphsTuple, mlip.CalcStresses(energy_fn=energy_fn).apply({}, batch))
    virials = np.asarray(cast(dict[str, Any], out.globals)[mlip.keys.predicted(mlip.keys.VIRIAL)])

    np.testing.assert_allclose(virials[0], expected, atol=1e-5)
    np.testing.assert_allclose(virials[1], expected, atol=1e-5)
