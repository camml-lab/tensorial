# -*- coding: utf-8 -*-
import jax.numpy as jnp
import jraph
import numpy as np
import optax
import pytest

from tensorial.gcnn import losses

NUM_GRAPHS = 10
NUM_NODES = 8
ENERGY_PREDICTIONS = np.random.rand(NUM_GRAPHS)
ENERGY_TARGETS = np.random.rand(NUM_GRAPHS)
FORCE_PREDICTIONS = np.random.rand(NUM_GRAPHS, NUM_NODES, 3)
FORCE_TARGETS = np.random.rand(NUM_GRAPHS, NUM_NODES, 3)

# pylint: disable=redefined-outer-name


@pytest.fixture
def graph_batch() -> jraph.GraphsTuple:
    graphs = []
    for energy_prediction, energy_target, force_predictions, force_targets in zip(
        ENERGY_PREDICTIONS, ENERGY_TARGETS, FORCE_PREDICTIONS, FORCE_PREDICTIONS
    ):
        graph_globals = {'energy': jnp.array([energy_target]), 'energy_prediction': jnp.array([energy_prediction])}
        graphs.append(
            jraph.GraphsTuple(
                nodes={
                    'forces': force_targets,
                    'force_predictions': force_predictions
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


def test_loss(graph_batch: jraph.GraphsTuple):
    optax_loss = optax.squared_error
    loss_fn = losses.Loss('globals.energy_prediction', 'globals.energy', loss_fn=optax_loss)

    loss = loss_fn(graph_batch)
    assert loss == optax_loss(graph_batch.globals['energy_prediction'], graph_batch.globals['energy']).mean()


def test_weighted_loss(graph_batch: jraph.GraphsTuple):
    optax_loss = optax.squared_error
    loss_fns = [(1., losses.Loss('globals.energy_prediction', 'globals.energy', loss_fn=optax_loss)),
                (10., losses.Loss('nodes.force_predictions', 'nodes.forces'))]

    loss_fn = losses.WeightedLoss(loss_fns)

    loss = loss_fn(graph_batch)
    total_loss = 0.
    for weight, loss_fn in loss_fns:
        total_loss += weight * loss_fn(graph_batch)
    assert loss == total_loss
