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
    trainer.fit(module, train_dataloaders=dataset, val_dataloaders=dataset)

    assert (pathlib.Path(trainer.log_dir) / "plots" / "train.pdf").exists()
    assert (pathlib.Path(trainer.log_dir) / "plots" / "validation.pdf").exists()


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

    plotter = reaxkit.GraphParityPlotter("globals.energy")
    trainer = reax.Trainer(default_root_dir=tmp_path, listeners=plotter)

    loss_fn = gcnn.losses.Loss(optax.losses.l2_loss, "globals.predicted_energy", "globals.energy")

    dataset = gcnn.data.GraphLoader([cube_graph])
    module = reaxkit.ReaxModule(
        Energy(),
        loss_fn=loss_fn,
        optimizer=optax.adamw(learning_rate=0.01),
        output=["predictions", "targets"],
    )
    trainer.fit(module, train_dataloaders=dataset, val_dataloaders=dataset)

    assert (pathlib.Path(trainer.log_dir) / "plots" / "train.pdf").exists()
    assert (pathlib.Path(trainer.log_dir) / "plots" / "validation.pdf").exists()
