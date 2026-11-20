import pathlib

from flax import linen
import jax.numpy as jnp
import jraph
import numpy as np
import optax
import pytest
import reax

from tensorial import base, gcnn, reaxkit
from tensorial.gcnn import keys
from tensorial.gcnn.atomic import keys as atomic_keys


class DummyModel(linen.Module):
    @linen.compact
    def __call__(self, x):
        return x**2


def test_parity_plotter(tmp_path):
    module = reaxkit.ReaxModule(
        DummyModel(),
        loss_fn=lambda x, y: optax.l2_loss(x, y).sum(),
        optimizer=optax.adamw(learning_rate=0.01),
        output=["predictions", "targets"],
    )

    plotter = reaxkit.ParityPlotter()
    trainer = reax.Trainer(default_root_dir=tmp_path, listeners=plotter)

    dataset = np.random.rand(2, 10)
    trainer.fit(module, train_dataloaders=dataset, val_dataloaders=dataset, max_epochs=1)

    figures = pathlib.Path(trainer.logger.log_dir) / "figures" / "parity"
    assert len(list((figures / "train").glob("step_*.png"))) == 1
    assert len(list((figures / "validation").glob("step_*.png"))) == 1


def test_graph_parity_plotter(cube_graph: jraph.GraphsTuple, tmp_path):
    cube_graph.globals[atomic_keys.TOTAL_ENERGY] = (
        jnp.linalg.norm(base.as_array(cube_graph.edges[keys.EDGE_LENGTHS])).sum().reshape(1, -1)
    )

    class Energy(linen.Module):
        @linen.compact
        def __call__(self, graph):
            energy = jnp.linalg.norm(base.as_array(cube_graph.edges[keys.EDGE_LENGTHS])).sum()
            globals = graph.globals
            globals[keys.predicted(atomic_keys.TOTAL_ENERGY)] = energy.reshape(1, -1)
            return graph._replace(globals=globals)

    plotter = reaxkit.GraphParityPlotter("globals.energy", fit_plot_every=5)
    trainer = reax.Trainer(default_root_dir=tmp_path, listeners=plotter)

    loss_fn = gcnn.losses.Loss(optax.losses.l2_loss, "globals.predicted_energy", "globals.energy")

    dataset = gcnn.data.GraphLoader([cube_graph])
    module = reaxkit.ReaxModule(
        Energy(),
        loss_fn=loss_fn,
        optimizer=optax.adamw(learning_rate=0.01),
        output=["predictions", "targets"],
    )
    trainer.fit(module, train_dataloaders=dataset, val_dataloaders=dataset, max_epochs=10)

    # Epochs 0, 4 and 9 are collected (fit_plot_every=5) and plotted as they end, with one batch
    # per epoch that is at updates 1, 5 and 10
    figures = pathlib.Path(trainer.logger.log_dir) / "figures" / "parity" / "energy"
    expected = ["step_00000001.png", "step_00000005.png", "step_00000010.png"]
    assert sorted(path.name for path in (figures / "train").iterdir()) == expected
    assert sorted(path.name for path in (figures / "validation").iterdir()) == expected


def test_graph_parity_plotter_quantities():
    # A single path, a tuple being the nodes of one path
    for targets in ("globals.energy", ("globals", "energy")):
        assert reaxkit.GraphParityPlotter(targets)._quantities == {
            "energy": (("globals", "energy"), ("globals", "predicted_energy"))
        }

    # Several quantities, named after the last path node unless that is ambiguous
    plotter = reaxkit.GraphParityPlotter(["globals.energy", "nodes.forces", "nodes.energy"])
    assert plotter._quantities == {
        "globals.energy": (("globals", "energy"), ("globals", "predicted_energy")),
        "forces": (("nodes", "forces"), ("nodes", "predicted_forces")),
        "nodes.energy": (("nodes", "energy"), ("nodes", "predicted_energy")),
    }

    # A mapping gives custom prediction paths
    plotter = reaxkit.GraphParityPlotter({"globals.energy": "globals.e", "nodes.forces": None})
    assert plotter._quantities == {
        "energy": (("globals", "energy"), ("globals", "e")),
        "forces": (("nodes", "forces"), ("nodes", "predicted_forces")),
    }

    with pytest.raises(ValueError):
        reaxkit.GraphParityPlotter(["globals.energy"], predictions="globals.e")
    with pytest.raises(ValueError):
        reaxkit.GraphParityPlotter([])
    with pytest.raises(ValueError):
        reaxkit.GraphParityPlotter(["globals.energy", "globals.energy"])


def _mlip_graphs():
    """Two real graphs (two and one nodes) and a padding graph (one node).

    The second graph's stress is a placeholder, marked by its per-field mask.
    """
    stress = np.arange(27.0).reshape(3, 3, 3)
    forces = np.arange(12.0).reshape(4, 3)
    targets = jraph.GraphsTuple(
        nodes={"forces": forces, "mask": np.array([True, True, True, False])},
        edges={},
        senders=np.array([], dtype=int),
        receivers=np.array([], dtype=int),
        globals={
            "energy": np.array([1.0, 2.0, 0.0]),
            "stress": stress,
            keys.mask_for("stress"): np.array([True, False, True]),
            "mask": np.array([True, True, False]),
        },
        n_node=np.array([2, 1, 1]),
        n_edge=np.array([0, 0, 0]),
    )
    predictions = targets._replace(
        nodes={**targets.nodes, keys.predicted("forces"): -forces},
        globals={
            **targets.globals,
            keys.predicted("energy"): -targets.globals["energy"],
            keys.predicted("stress"): -stress,
        },
    )
    return targets, predictions


def test_graph_parity_plotter_masks_and_optional():
    targets, predictions = _mlip_graphs()

    plotter = reaxkit.GraphParityPlotter(
        ["globals.energy", "nodes.forces", "globals.stress", "globals.dipole"], optional=True
    )
    plotter._collect_batch_data("test", predictions, (targets, targets))

    store = plotter.data_store["test"]
    # The dipole isn't in the data, so it is skipped
    assert list(store) == [("energy", None), ("forces", None), ("stress", None)]

    # The level's padding mask is applied
    (energy,), (predicted_energy,) = store[("energy", None)]
    np.testing.assert_allclose(energy, [1.0, 2.0])
    np.testing.assert_allclose(predicted_energy, -energy)
    (forces,), _ = store[("forces", None)]
    np.testing.assert_allclose(forces, np.arange(9.0))
    # ...and the per-field mask, which drops the second graph's placeholder stress
    (stress,), _ = store[("stress", None)]
    np.testing.assert_allclose(stress, np.arange(9.0))

    # Without `optional`, a missing quantity is an error
    plotter = reaxkit.GraphParityPlotter(["globals.energy", "globals.dipole"])
    with pytest.raises(KeyError):
        plotter._collect_batch_data("test", predictions, (targets, targets))


def test_graph_parity_plotter_logs_each_quantity(tmp_path):
    targets, predictions = _mlip_graphs()
    plotter = reaxkit.GraphParityPlotter(
        ["globals.energy", "nodes.forces"], plot_final_combined=True
    )
    plotter._collect_batch_data("test", predictions, (targets, targets))

    trainer = reax.Trainer(default_root_dir=tmp_path)
    plotter._plot_parity(trainer, "test")
    plotter._plot_combined_all_stages(trainer)

    figures = pathlib.Path(trainer.logger.log_dir) / "figures" / "parity"
    for quantity in ("energy", "forces"):
        assert (figures / quantity / "test.png").exists()
        assert (figures / quantity / "combined.png").exists()


def test_irreps_graph_parity_plotter(tmp_path):
    import e3nn_jax as e3j

    from tensorial.reaxkit.listeners.parity_plotter import IrrepsGraphParityPlotter

    # Mock IrrepsArray data
    irreps = e3j.Irreps("1x0e + 1x1o")
    tensor = e3j.IrrepsArray(irreps, jnp.array([[2.0, 1.0, 0.0, -1.0]]))

    plotter = IrrepsGraphParityPlotter(
        targets="nodes.tensors",
    )

    # Manually check decomposition
    decomposed = plotter._decompose_data(tensor)

    assert "0e" in decomposed
    assert "1o" in decomposed

    # Expected values
    np.testing.assert_allclose(decomposed["0e"], 2.0)
    np.testing.assert_allclose(decomposed["1o"], jnp.array([[1.0, 0.0, -1.0]]))


def test_irreps_graph_parity_plotter_repeated_irreps():
    import e3nn_jax as e3j

    tensor = e3j.IrrepsArray("2x0e + 1x1o + 1x0e", jnp.arange(6.0).reshape(1, 6))
    plotter = reaxkit.listeners.IrrepsGraphParityPlotter(targets="nodes.tensors")

    decomposed = plotter._decompose_data(tensor)

    # Each segment keeps only its own channels, even when an irrep is repeated
    np.testing.assert_allclose(decomposed["0e"], [[0.0, 1.0]])
    np.testing.assert_allclose(decomposed["1o"], [[2.0, 3.0, 4.0]])
    np.testing.assert_allclose(decomposed["0e_2"], [[5.0]])


def test_irreps_graph_parity_plotter_masked_nodes():
    import e3nn_jax as e3j

    irreps = e3j.Irreps("0e + 1o")
    values = jnp.arange(12.0).reshape(3, 4)
    mask = jnp.array([True, False, True])
    targets = jraph.GraphsTuple(
        nodes={"tensors": e3j.IrrepsArray(irreps, values), "mask": mask},
        edges=None,
        senders=None,
        receivers=None,
        globals=None,
        n_node=jnp.array([3]),
        n_edge=jnp.array([0]),
    )
    predictions = targets._replace(
        nodes={**targets.nodes, keys.predicted("tensors"): e3j.IrrepsArray(irreps, -values)}
    )

    plotter = reaxkit.listeners.IrrepsGraphParityPlotter(targets="nodes.tensors")
    plotter._collect_batch_data("test", predictions, (targets, targets))

    store = plotter.data_store["test"]
    # The mask is applied without losing the irreps, so every irrep gets its own entry
    assert list(store) == [("tensors", "0e"), ("tensors", "1o")]
    (gt_0e,), _ = store[("tensors", "0e")]
    (gt_1o,), (pred_1o,) = store[("tensors", "1o")]
    np.testing.assert_allclose(gt_0e, [0.0, 8.0])
    np.testing.assert_allclose(gt_1o, [1.0, 2.0, 3.0, 9.0, 10.0, 11.0])
    np.testing.assert_allclose(pred_1o, -gt_1o)
