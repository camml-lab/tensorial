import pathlib

from flax import linen
import jax.numpy as jnp
import jraph
import numpy as np
import optax
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

    assert (pathlib.Path(trainer.log_dir) / "plots" / "train_epoch_0.pdf").exists()
    assert (pathlib.Path(trainer.log_dir) / "plots" / "validation_epoch_0.pdf").exists()


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

    assert (pathlib.Path(trainer.log_dir) / "plots" / "train_epoch_9.pdf").exists()
    assert (pathlib.Path(trainer.log_dir) / "plots" / "validation_epoch_9.pdf").exists()


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

    (gt,), (pred,) = plotter.data_store["test"]
    # The mask is applied without losing the irreps, so every irrep gets its own entry
    np.testing.assert_allclose(gt["0e"], [[0.0], [8.0]])
    np.testing.assert_allclose(gt["1o"], [[1.0, 2.0, 3.0], [9.0, 10.0, 11.0]])
    np.testing.assert_allclose(pred["1o"], -gt["1o"])
