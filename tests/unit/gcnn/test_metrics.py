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
from tensorial.gcnn import keys
from tensorial.gcnn.metrics._base import mdiv


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
    num_graphs: Final[int] = 13
    num_nodes: Final[int] = 100
    type_fields: Final[str] = "type_id"
    num_types: Final[int] = 3

    random_graphs = gcnn.random.spatial_graph(
        rng_key,
        cutoff=0.2,
        num_graphs=num_graphs,
        num_nodes=num_nodes,
        nodes={
            type_fields: lambda rng_key, num: jax.random.randint(
                rng_key, shape=(num, 1), minval=0, maxval=num_types
            ),
            gcnn.keys.MASK: (
                lambda rng_key, num: jax.random.randint(
                    rng_key, shape=(num,), minval=0, maxval=2
                ).astype(bool)
            ),
        },
    )

    node_types = list(range(num_types))
    # Shuffle to make sure this metric works with type list that isn't ordered
    random.shuffle(node_types)
    avg_num_neighbours = gcnn.metrics.AvgNumNeighboursByType(node_types, type_field=type_fields)

    loader = gcnn.data.GraphLoader(random_graphs, batch_size=batch_size)

    trainer = reax.Trainer()
    logged: dict = trainer.eval_stats(avg_num_neighbours, loader).logged_metrics
    res: dict[int, jt.Float[jax.Array, "n_types"]] = logged[
        gcnn.metrics.AvgNumNeighboursByType.__name__
    ]

    all_graphs = jraph.batch(random_graphs)
    counts = jnp.bincount(all_graphs.senders, length=all_graphs.n_node.sum().item())

    for i in range(num_types):
        # Get all valid nodes of the right type
        mask = all_graphs.nodes[gcnn.keys.MASK] & (all_graphs.nodes[type_fields][:, 0] == i)
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


def _graphs(**node_fields):
    """Build a single 4-node graph with two types and three edges."""
    default = {
        "type_id": jnp.array([[1], [2], [1], [2]]),
        keys.MASK: jnp.array([True, True, True, False]),
    }
    default.update(node_fields)
    return jraph.GraphsTuple(
        n_node=jnp.array([4]),
        n_edge=jnp.array([3]),
        nodes=default,
        edges={},
        globals={},
        senders=jnp.array([0, 0, 1]),
        receivers=jnp.array([1, 3, 2]),
    )


def test_mdiv_happy_path():
    num = jnp.array([1.0, 2.0])
    denom = jnp.array([2.0, 4.0])
    assert jnp.allclose(mdiv(num, denom), jnp.array([0.5, 0.5]))


def test_mdiv_with_mask():
    # Where the mask is False the denominator is forced to 1, so the numerator is returned
    num = jnp.array([1.0, 2.0])
    denom = jnp.array([2.0, 0.0])
    where = jnp.array([True, False])
    result = mdiv(num, denom, where=where)
    assert jnp.allclose(result, jnp.array([0.5, 2.0]))


def test_mdiv_shape_mismatch_raises():
    with pytest.raises(ValueError):
        mdiv(jnp.array([1.0, 2.0]), jnp.array([1.0, 2.0, 3.0]))


def test_graph_metric_metric_property():
    """The ``metric`` property exposes the stored parent metric instance."""
    metric = gcnn.metrics.graph_metric(reax.metrics.Sum, predictions="nodes.pred", mask=None)

    assert metric.is_empty
    graph = _graphs(pred=jnp.array([1.0, 2.0, 3.0, 4.0]))
    created = metric.create(graph)

    assert not created.is_empty
    assert isinstance(created.metric, reax.metrics.Sum)


def test_graph_metric_reduce_and_empty_raises():
    metric = gcnn.metrics.graph_metric(reax.metrics.Sum, predictions="nodes.pred", mask=None)
    graph = _graphs(pred=jnp.array([1.0, 2.0, 3.0, 4.0]))

    # An empty metric cannot be computed or reduced
    with pytest.raises(RuntimeError):
        metric.compute()
    with pytest.raises(RuntimeError):
        metric.reduce(axis=0)

    # A populated one can
    created = metric.create(graph)
    assert created.compute() == 10.0
    reduced = created.reduce(axis=0)
    assert reduced.compute() == 10.0


def test_graph_metric_merge():
    metric = gcnn.metrics.graph_metric(reax.metrics.Sum, predictions="nodes.pred", mask=None)
    left = metric.create(_graphs(pred=jnp.array([1.0, 1.0, 1.0, 1.0])))
    right = metric.create(_graphs(pred=jnp.array([2.0, 2.0, 2.0, 2.0])))

    # Each graph contributes 4 + 8 = 12, so merging both gives 24
    assert left.merge(right).compute() == 12.0

    # Merging with an empty metric returns the non-empty one
    assert left.merge(metric).compute() == 4.0


def test_graph_metric_normalise_by_prediction():
    # ``normalise_by`` divides the prediction before accumulation
    metric = gcnn.metrics.graph_metric(
        reax.metrics.Sum, predictions="nodes.pred", normalise_by="nodes.norm", mask=None
    )
    graph = _graphs(pred=jnp.array([2.0, 4.0, 6.0, 8.0]), norm=jnp.array([2.0, 2.0, 2.0, 2.0]))

    # [2,4,6,8] / [2,2,2,2] -> [1,2,3,4]; sum = 10
    assert metric.create(graph).compute() == 10.0


def test_graph_metric_normalise_by_target():
    metric = gcnn.metrics.graph_metric(
        reax.metrics.MeanSquaredError,
        predictions="nodes.pred",
        targets="nodes.targ",
        normalise_by="nodes.norm",
    )
    # preds [4,6,8,10] / 2 -> [2,3,4,5]; targs [0,2,4,6] / 2 -> [0,1,2,3]; mean((2,2,2,2)^2) = 4
    graph = _graphs(
        pred=jnp.array([4.0, 6.0, 8.0, 10.0]),
        targ=jnp.array([0.0, 2.0, 4.0, 6.0]),
        norm=jnp.array([2.0, 2.0, 2.0, 2.0]),
    )

    assert np.isclose(metric.update(graph).compute(), 4.0)


def test_avg_num_neighbours_by_type_compute():
    metric = gcnn.metrics.AvgNumNeighboursByType([1, 2])
    result = metric.create(_graphs()).compute()  # pylint: disable=not-callable

    assert set(result) == {1, 2}
    assert result[1] == 1.0  # node 0 has two edges, node 2 has none
    assert result[2] == 1.0  # node 1 has one edge, node 3 is masked out


def test_avg_num_neighbours_by_type_update_empty_shortcut():
    """When the metric is empty, ``update()`` short-circuits to ``create()``."""
    metric = gcnn.metrics.AvgNumNeighboursByType([1, 2])

    updated = metric.update(_graphs())
    assert not updated.is_empty
    assert updated.compute() == {1: 1.0, 2: 1.0}


def test_avg_num_neighbours_by_type_update_nonempty():
    """When the metric already has state, ``update()`` merges via each ``Average``'s ``merge``."""
    metric = gcnn.metrics.AvgNumNeighboursByType([1, 2])
    created = metric.create(_graphs())

    updated = created.update(_graphs())
    assert not updated.is_empty
    assert updated.compute() == {1: 1.0, 2: 1.0}


def test_avg_num_neighbours_by_type_empty_idempotent():
    metric = gcnn.metrics.AvgNumNeighboursByType([1, 2])

    assert metric.empty() is metric


def test_avg_num_neighbours_by_type_empty_resets_state():
    """``empty()`` on a metric with state returns a fresh, empty metric of the same type map."""
    populated = gcnn.metrics.AvgNumNeighboursByType([1, 2]).create(_graphs())

    emptied = populated.empty()
    assert emptied is not populated
    assert emptied.is_empty
    # The type map is preserved
    assert emptied._node_types.tolist() == [1, 2]  # pylint: disable=protected-access


def test_avg_num_neighbours_by_type_merge_type_mismatch_raises():
    """Merging metrics whose type maps differ must raise ``ValueError`` (not a JAX broadcast error)."""
    left = gcnn.metrics.AvgNumNeighboursByType([1, 2])
    for right_types in ([1, 5], [1, 2, 3]):
        with pytest.raises(ValueError, match="Type maps must match"):
            left.create(_graphs()).merge(
                gcnn.metrics.AvgNumNeighboursByType(right_types).create(_graphs())
            )


def test_avg_num_neighbours_by_type_merge_empty():
    metric_cls = gcnn.metrics.AvgNumNeighboursByType
    populated = metric_cls([1, 2]).create(_graphs())
    empty = metric_cls([1, 2])

    # other is empty -> returns self
    assert populated.merge(empty) == populated
    # self is empty -> returns other
    assert empty.merge(populated) == populated


def test_avg_num_neighbours_by_type_compute_empty_raises():
    metric = gcnn.metrics.AvgNumNeighboursByType([1, 2])

    with pytest.raises(RuntimeError):
        metric.compute()  # pylint: disable=not-callable
