# from typing import Literal

import jax
import jax.numpy as jnp

# import jaxtyping as jt
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

    forces_loss = gcnn.graph_ops.segment_reduce(
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


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_loss_reports_its_reduction(reduction):
    """Anything aggregating losses over batches needs to know how they were reduced"""
    loss_fn = losses.Loss(
        optax.squared_error, "globals.energy", "globals.energy_prediction", reduction=reduction
    )
    assert loss_fn.reduction == reduction


def test_weighted_loss_reduction():
    optax_loss = optax.squared_error

    def make(reduction):
        return losses.Loss(
            optax_loss, "globals.energy", "globals.energy_prediction", reduction=reduction
        )

    assert losses.WeightedLoss([make("sum"), make("sum")]).reduction == "sum"
    assert losses.WeightedLoss([make("mean"), make("mean")]).reduction == "mean"
    # Terms that disagree don't add up to either one, so we fall back to the default reduction
    assert losses.WeightedLoss([make("sum"), make("mean")]).reduction == "mean"


def num_nodes(graph: jraph.GraphsTuple) -> int:
    return sum(graph.n_node)


def num_edges(graph: jraph.GraphsTuple) -> int:
    return sum(graph.n_edge)


def num_graphs(graph: jraph.GraphsTuple) -> int:
    return len(graph.n_node)


def test_losses_as_jit_arguments(graph_batch: jraph.GraphsTuple):
    """Losses are pytrees with static configuration, so they can be passed into jitted functions"""
    energy = losses.Loss(optax.squared_error, "globals.energy_prediction", "globals.energy")
    forces = losses.Loss(
        optax.squared_error,
        "nodes.force_predictions",
        "nodes.forces",
        reduction="sum",
        mask_field="nodes.mask",
    )
    weighted = losses.WeightedLoss([energy, forces], [0.5, 2.0])

    @jax.jit
    def evaluate(loss_fn: losses.GraphLoss, graphs: jraph.GraphsTuple) -> jax.Array:
        return loss_fn(graphs)

    for loss_fn in (energy, forces, weighted):
        assert jnp.allclose(evaluate(loss_fn, graph_batch), loss_fn(graph_batch))


# ---- Optional terms and per-field masks ---------------------------------------


def _stress_batch(stress_mask=None) -> jraph.GraphsTuple:
    """Two graphs with energies and stresses (and, optionally, a per-field stress mask)."""
    graph_globals = {
        "energy": jnp.array([1.0, 2.0]),
        "predicted_energy": jnp.array([1.5, 2.0]),
        "stress": jnp.zeros((2, 3, 3)),
        "predicted_stress": jnp.stack([jnp.full((3, 3), 2.0), jnp.full((3, 3), 10.0)]),
    }
    if stress_mask is not None:
        graph_globals["stress_mask"] = jnp.array(stress_mask)
    return jraph.GraphsTuple(
        nodes={},
        edges={},
        receivers=jnp.array([], dtype=int),
        senders=jnp.array([], dtype=int),
        globals=graph_globals,
        n_node=jnp.array([0, 0]),
        n_edge=jnp.array([0, 0]),
    )


def _without_stress(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    return graph._replace(globals={k: v for k, v in graph.globals.items() if "stress" not in k})


def test_field_mask_is_applied():
    loss_fn = losses.Loss("squared_error", "globals.stress", "globals.predicted_stress")
    # Unmasked: mean over both graphs of the (constant) squared errors
    assert loss_fn(_stress_batch()) == pytest.approx((4.0 + 100.0) / 2)
    # The second graph's stress is a placeholder, so only the first one counts
    assert loss_fn(_stress_batch(stress_mask=[True, False])) == pytest.approx(4.0)


@pytest.mark.parametrize("jit", [True, False])
def test_optional_term_is_skipped_when_missing(jit):
    energy = losses.Loss("squared_error", "globals.energy", "globals.predicted_energy")
    stress = losses.Loss(
        "squared_error", "globals.stress", "globals.predicted_stress", optional=True
    )
    weighted = losses.WeightedLoss([energy, stress], [1.0, 3.0])
    if jit:
        weighted = jax.jit(weighted)

    graph = _stress_batch()
    assert weighted(graph) == pytest.approx(0.25 / 2 + 3.0 * 52.0)
    # Without stresses, only the energy term is left
    assert weighted(_without_stress(graph)) == pytest.approx(0.25 / 2)


def test_required_term_missing_raises():
    stress = losses.Loss("squared_error", "globals.stress", "globals.predicted_stress")
    energy = losses.Loss("squared_error", "globals.energy", "globals.predicted_energy")
    with pytest.raises(KeyError):
        losses.WeightedLoss([energy, stress])(_without_stress(_stress_batch()))


def test_all_terms_missing_raises():
    stress = losses.Loss(
        "squared_error", "globals.stress", "globals.predicted_stress", optional=True
    )
    with pytest.raises(ValueError, match="None of the loss terms"):
        losses.WeightedLoss([stress])(_without_stress(_stress_batch()))
    # On its own, an optional loss with no data can't be skipped
    with pytest.raises(ValueError, match="no data in this batch"):
        stress(_without_stress(_stress_batch()))


def test_loss_with_contributions_leaves_out_inactive_terms():
    energy = losses.Loss("squared_error", "globals.energy", "globals.predicted_energy")
    stress = losses.Loss(
        "squared_error", "globals.stress", "globals.predicted_stress", optional=True
    )
    total, contribs = losses.WeightedLoss([energy, stress]).loss_with_contributions(
        _without_stress(_stress_batch()), _without_stress(_stress_batch())
    )
    assert list(contribs) == [energy.label()]
    assert total == pytest.approx(0.25 / 2)


def test_fully_masked_global_loss_is_zero():
    # e.g. a batch of only molecules in a dataset that also has crystals with stresses
    loss_fn = losses.Loss("squared_error", "globals.stress", "globals.predicted_stress")
    assert loss_fn(_stress_batch(stress_mask=[False, False])) == pytest.approx(0.0)


# ---- normalise_by -------------------------------------------------------------


def _energy_graphs() -> jraph.GraphsTuple:
    """Two graphs with 2 and 4 nodes whose total energies are off by 1 and 2."""
    graphs = [
        jraph.GraphsTuple(
            nodes={"x": jnp.zeros((n_node, 1))},
            edges={},
            receivers=jnp.array([], dtype=int),
            senders=jnp.array([], dtype=int),
            globals={"energy": jnp.array([0.0]), "predicted_energy": jnp.array([error])},
            n_node=jnp.array([n_node]),
            n_edge=jnp.array([0]),
        )
        for n_node, error in ((2, 1.0), (4, 2.0))
    ]
    return jraph.batch(graphs)


def test_normalise_by_n_node():
    loss_fn = losses.Loss(
        "squared_error", "globals.energy", "globals.predicted_energy", normalise_by="n_node"
    )
    # Per-atom errors are 1/2 and 2/4
    assert loss_fn(_energy_graphs()) == pytest.approx(0.25)


@pytest.mark.parametrize("jit", [True, False])
def test_normalise_by_ignores_padding(jit):
    """Padding nodes go in separate (empty or masked) graphs, so the loss is unchanged."""
    loss_fn = losses.Loss(
        "squared_error", "globals.energy", "globals.predicted_energy", normalise_by="n_node"
    )
    if jit:
        loss_fn = jax.jit(loss_fn)
    graph = _energy_graphs()
    padded = gcnn.data.pad_with_graphs(graph, n_node=9, n_edge=1, n_graph=4)
    assert int(padded.n_node[-1]) == 0  # an empty padding graph, so would divide by zero

    assert loss_fn(padded) == pytest.approx(loss_fn(graph))


def test_normalise_by_requires_global_target():
    with pytest.raises(ValueError, match="global target"):
        losses.Loss("squared_error", "nodes.forces", normalise_by="n_node")
