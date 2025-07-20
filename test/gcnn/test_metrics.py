import random
from typing import Final

import jax
import jax.numpy as jnp
import jaxtyping as jt
import jraph
import numpy as np
import optax
import pytest
import reax

from tensorial import gcnn


@pytest.mark.parametrize("mask_field", [None, "auto"])
def test_graph_metric(mask_field):
    N_GRAPHS: Final[int] = 10
    N_NODES: Final[int] = 4

    targets = np.random.random((N_GRAPHS, N_NODES))
    preds = np.random.random((N_GRAPHS, N_NODES))
    graphs = []
    for target, pred in zip(targets, preds):
        graphs.append(
            jraph.GraphsTuple(
                n_node=jnp.array([N_NODES]),
                n_edge=jnp.zeros(1),
                nodes={"target": target, "pred": pred},
                edges={},
                globals={},
                senders=jnp.array([]),
                receivers=jnp.array([]),
            )
        )
    graphs = jraph.batch(graphs)
    graph_metrics = gcnn.metrics.graph_metric(
        reax.metrics.MeanSquaredError,
        predictions="nodes.pred",
        targets="nodes.target",
        mask=mask_field,
    ).update(graphs)
    reference = optax.losses.squared_error(preds, targets).mean()
    computed = graph_metrics.compute()
    assert np.isclose(computed, reference)


@pytest.mark.parametrize("mask_field", ["nodes.mask", "auto"])
def test_graph_metric_with_mask(mask_field):
    N_GRAPHS: Final[int] = 10
    N_NODES: Final[int] = 4

    targets = np.random.random((N_GRAPHS, N_NODES, 3))
    preds = np.random.random((N_GRAPHS, N_NODES, 3))
    masks = np.random.randint(0, 2, size=(N_GRAPHS, N_NODES), dtype=bool)
    graphs = []
    for target, pred, mask in zip(targets, preds, masks):
        graphs.append(
            jraph.GraphsTuple(
                n_node=jnp.array([N_NODES]),
                n_edge=jnp.zeros(1),
                nodes={"target": target, "pred": pred, "mask": mask},
                edges={},
                globals={},
                senders=jnp.array([]),
                receivers=jnp.array([]),
            )
        )
    del target, pred, mask
    graphs = jraph.batch(graphs)

    # Manual mask
    graph_metrics = gcnn.metrics.graph_metric(
        reax.metrics.MeanSquaredError,
        predictions="nodes.pred",
        targets="nodes.target",
        mask=mask_field,
    ).update(graphs)
    # masks = masks.reshape(N_GRAPHS, N_NODES, np.newaxis)
    reference = optax.losses.squared_error(preds[masks], targets[masks]).mean()
    computed = graph_metrics.compute()
    assert np.isclose(computed, reference)


@pytest.mark.parametrize("batch_size", [1, 3, 100])
def test_indexed_metrics(rng_key, batch_size: int):
    NUM_GRAPHS: Final[int] = 13
    NUM_NODES: Final[int] = 100
    TYPE_FIELD: Final[str] = "type_id"
    NUM_TYPES: Final[int] = 3

    random_graphs = gcnn.random.spatial_graph(
        rng_key,
        cutoff=0.2,
        num_graphs=NUM_GRAPHS,
        num_nodes=NUM_NODES,
        nodes={
            TYPE_FIELD: lambda rng_key, num: jax.random.randint(
                rng_key, shape=(num,), minval=0, maxval=NUM_TYPES
            ),
            gcnn.keys.MASK: (
                lambda rng_key, num: jax.random.randint(
                    rng_key, shape=(num,), minval=0, maxval=2
                ).astype(bool)
            ),
        },
    )

    node_types = list(range(NUM_TYPES))
    # Shuffle to make sure this metric works with type list that isn't ordered
    random.shuffle(node_types)
    avg_num_neighbours = gcnn.metrics.AvgNumNeighboursByType(node_types, type_field=TYPE_FIELD)

    loader = gcnn.data.GraphLoader(random_graphs, batch_size=batch_size)

    trainer = reax.Trainer()
    res: dict[int, jt.Float[jax.Array, "n_types"]] = trainer.eval_stats(
        avg_num_neighbours, loader
    ).logged_metrics[gcnn.metrics.AvgNumNeighboursByType.__name__]

    all_graphs = jraph.batch(random_graphs)
    counts = jnp.bincount(all_graphs.senders, length=all_graphs.n_node.sum().item())

    for i in range(NUM_TYPES):
        # Get all valid nodes of the right type
        mask = all_graphs.nodes[gcnn.keys.MASK] & (all_graphs.nodes[TYPE_FIELD] == i)
        assert jnp.isclose(counts[mask].mean(), res[i])


def test_metrics_registry():
    """Test that the metrics are correctly picked up through the plugin system"""
    expected = {
        "atomic/num_species": gcnn.atomic.NumSpecies,
        "atomic/all_atomic_numbers": gcnn.atomic.AllAtomicNumbers,
        "atomic/avg_num_neighbours": gcnn.atomic.AvgNumNeighbours,
        "atomic/force_std": gcnn.atomic.ForceStd,
        "atomic/energy_per_atom_lstsq": gcnn.atomic.EnergyPerAtomLstsq,
    }

    registry = reax.metrics.get_registry()

    for metric_name in expected:
        assert metric_name in registry
