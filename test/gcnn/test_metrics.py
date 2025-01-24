import jax.numpy as jnp
import jraph
import numpy as np
import optax
import pytest
import reax

from tensorial.gcnn import metrics


@pytest.mark.parametrize("mask_field", [None, "auto"])
def test_graph_metric(mask_field):
    N_GRAPHS = 10
    N_NODES = 4

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
    graph_metrics = metrics.graph_metric(
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
    N_GRAPHS = 10
    N_NODES = 4

    targets = np.random.random((N_GRAPHS, N_NODES, 3))
    preds = np.random.random((N_GRAPHS, N_NODES, 3))
    masks = np.random.randint(0, 2, size=(N_GRAPHS, N_NODES), dtype=np.bool)
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
    graph_metrics = metrics.graph_metric(
        reax.metrics.MeanSquaredError,
        predictions="nodes.pred",
        targets="nodes.target",
        mask=mask_field,
    ).update(graphs)
    # masks = masks.reshape(N_GRAPHS, N_NODES, np.newaxis)
    reference = optax.losses.squared_error(preds[masks], targets[masks]).mean()
    computed = graph_metrics.compute()
    assert np.isclose(computed, reference)
