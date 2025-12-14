from typing import Literal

import jax
import jax.numpy as jnp
import jaxtyping as jt
import jraph
import numpy as np
import optax
import pytest

from tensorial import gcnn
from tensorial.gcnn import losses

NUM_GRAPHS = 10
NUM_NODES = 8
ENERGY_PREDICTIONS = np.random.rand(NUM_GRAPHS)
ENERGY_TARGETS = np.random.rand(NUM_GRAPHS)
FORCE_PREDICTIONS = np.random.rand(NUM_GRAPHS, NUM_NODES, 3)
FORCE_TARGETS = np.random.rand(NUM_GRAPHS, NUM_NODES, 3)
FORCE_MASKS = np.random.choice(2, (NUM_GRAPHS, NUM_NODES)).astype(bool)

# pylint: disable=redefined-outer-name


@pytest.fixture
def graph_batch() -> jraph.GraphsTuple:
    graphs = []
    for i in range(NUM_GRAPHS):
        graph_globals = {
            "energy": jnp.array([ENERGY_TARGETS[i]]),
            "energy_prediction": jnp.array([ENERGY_PREDICTIONS[i]]),
        }
        graphs.append(
            jraph.GraphsTuple(
                nodes={
                    "forces": jnp.array(FORCE_TARGETS[i]),
                    "force_predictions": jnp.array(FORCE_PREDICTIONS[i]),
                    "mask": jnp.array(FORCE_MASKS[i]),
                },
                edges={},
                receivers=jnp.array([]),
                senders=jnp.array([]),
                globals=graph_globals,
                n_node=jnp.array([NUM_NODES]),
                n_edge=jnp.array([NUM_NODES]),
            )
        )

    return jraph.batch(graphs)


@pytest.mark.parametrize("jit", [True, False])
def test_loss(jit, graph_batch: jraph.GraphsTuple):
    optax_loss = optax.squared_error
    loss_fn = losses.Loss(optax_loss, "globals.energy_prediction", "globals.energy")
    if jit:
        loss_fn = jax.jit(loss_fn)

    loss = loss_fn(graph_batch)
    assert jnp.isclose(
        loss,
        optax_loss(graph_batch.globals["energy_prediction"], graph_batch.globals["energy"]).mean(),
    )


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("mask_field", [None, "nodes.mask"])
def test_masked_loss(jit, mask_field, graph_batch: jraph.GraphsTuple):
    optax_loss = optax.squared_error
    loss_fn = losses.Loss(
        optax_loss, "nodes.forces", "nodes.force_predictions", mask_field=mask_field
    )
    if jit:
        loss_fn = jax.jit(loss_fn)

    loss = loss_fn(graph_batch)

    forces = graph_batch.nodes["forces"]
    pred_forces = graph_batch.nodes["force_predictions"]

    forces_loss = gcnn.segment.reduce_masked(
        optax_loss(forces, pred_forces),
        graph_batch.n_node,
        reduction="mean",
        mask=graph_batch.nodes["mask"],
    )

    assert jnp.isclose(loss, forces_loss.mean())


@pytest.mark.parametrize("jit", [True, False])
@pytest.mark.parametrize("weights", [None, [1.0, 10.0]])
def test_weighted_loss(jit, weights, graph_batch: jraph.GraphsTuple):
    optax_loss = optax.squared_error
    loss_fns = [
        losses.Loss(optax_loss, "globals.energy", "globals.energy_prediction"),
        losses.Loss(optax_loss, "nodes.forces", "nodes.force_predictions"),
    ]

    loss_fn = losses.WeightedLoss(loss_fns, weights)
    if weights is None:
        assert jnp.allclose(loss_fn.weights, 1.0)

    if jit:
        fn = jax.jit(loss_fn)
    else:
        fn = loss_fn

    loss = fn(graph_batch)
    total_loss = jnp.dot(
        jnp.array(loss_fn.weights),
        jnp.array(list(loss_fn(graph_batch) for loss_fn in loss_fns)),
    )

    assert jnp.isclose(loss, total_loss)


@pytest.mark.parametrize("jit", [True, False])
def test_loss_with_padding(jit, graph_batch: jraph.GraphsTuple):
    padded = jraph.pad_with_graphs(
        graph_batch,
        num_nodes(graph_batch) + 1,
        num_edges(graph_batch) + 1,
        num_graphs(graph_batch) + 1,
    )
    padded = gcnn.data.add_padding_mask(padded)

    optax_loss = optax.squared_error
    loss_fn = losses.Loss(optax_loss, "globals.energy", "globals.energy_prediction")
    if jit:
        loss_fn = jax.jit(loss_fn)

    loss = loss_fn(graph_batch)
    padded_loss = loss_fn(padded)
    assert jnp.isclose(padded_loss, loss)  # Padding shouldn't change the loss value


def num_nodes(graph: jraph.GraphsTuple) -> int:
    return sum(graph.n_node)


def num_edges(graph: jraph.GraphsTuple) -> int:
    return sum(graph.n_edge)


def num_graphs(graph: jraph.GraphsTuple) -> int:
    return len(graph.n_node)
