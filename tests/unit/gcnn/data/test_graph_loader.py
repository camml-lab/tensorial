"""Tests for :class:`tensorial.gcnn.data.GraphLoader`."""

import math

import jax
import jax.numpy as jnp
import jraph
import numpy as np
import pytest
from reax.data import samplers

from tensorial import gcnn
from tensorial.gcnn import keys


def _trivial_graph(n_node: int = 2, n_edge: int = 3) -> jraph.GraphsTuple:
    """A single non-empty graph with one node/edge/global field."""
    return jraph.GraphsTuple(
        n_node=np.array([n_node]),
        n_edge=np.array([n_edge]),
        nodes={"f": np.ones((n_node, 1), dtype=np.float32)},
        edges={"f": np.ones((n_edge, 1), dtype=np.float32)},
        globals={"g": np.ones((1,), dtype=np.float32)},
        senders=np.zeros(n_edge, dtype=np.int32),
        receivers=np.zeros(n_edge, dtype=np.int32),
    )


def _n_graphs(n: int) -> list[jraph.GraphsTuple]:
    return [_trivial_graph() for _ in range(n)]


def _graphs_with_globals(n: int) -> list[jraph.GraphsTuple]:
    """Graphs each carrying a distinguishable global value ``g = i``."""
    return [
        jraph.GraphsTuple(
            n_node=np.array([2]),
            n_edge=np.array([3]),
            nodes={"f": np.ones((2, 1), dtype=np.float32)},
            edges={"f": np.ones((3, 1), dtype=np.float32)},
            globals={"g": np.array([i], dtype=np.float32)},
            senders=np.zeros(3, dtype=np.int32),
            receivers=np.zeros(3, dtype=np.int32),
        )
        for i in range(n)
    ]


# ---- Constructor / basic properties ---------------------------------------

def test_loader_stores_params_and_properties():
    graphs = _n_graphs(5)
    loader = gcnn.data.GraphLoader(graphs, batch_size=2, shuffle=False, pad=True)

    assert loader.batch_size == 2
    assert loader.shuffle is False
    # `dataset` is a tuple of positions; the single position holds the 5 unbatched graphs
    assert len(loader.dataset) == 1
    assert len(loader.dataset[0]) == 5
    # padding is exposed from the inner batcher
    assert loader.padding is not None
    assert loader.padding.n_graphs == 3  # batch_size + 1
    assert len(loader) == 3  # ceil(5 / 2)


def test_loader_passes_pad_and_padding_to_batcher():
    graphs = _n_graphs(4)
    loader = gcnn.data.GraphLoader(graphs, batch_size=2, pad=True)
    assert loader._batchers[0]._add_mask is True


def test_loader_passes_batch_mode_to_batcher():
    graphs = _n_graphs(4)
    explicit = gcnn.data.GraphLoader(graphs, batch_size=2, batch_mode=gcnn.data.BatchMode.EXPLICIT)
    implicit = gcnn.data.GraphLoader(graphs, batch_size=2, batch_mode=gcnn.data.BatchMode.IMPLICIT)
    assert str(explicit._batchers[0]._mode) == "BatchMode.EXPLICIT"
    assert str(implicit._batchers[0]._mode) == "BatchMode.IMPLICIT"


def test_loader_unbatches_batched_input_graphs():
    """A batched ``GraphsTuple`` input must be unbatched into individual graphs."""
    graphs = [_trivial_graph() for _ in range(4)]
    batched = jraph.batch_np(graphs)

    loader = gcnn.data.GraphLoader(batched, batch_size=2)
    assert len(loader.dataset[0]) == 4
    # Each unbatched entry has shape [1] on n_node
    for g in loader.dataset[0]:
        assert len(g.n_node) == 1


def test_loader_unbatches_multiple_positions():
    inputs = _n_graphs(6)
    targets = _n_graphs(6)
    loader = gcnn.data.GraphLoader(inputs, targets, batch_size=3)
    assert len(loader.dataset) == 2
    assert all(len(g.n_node) == 1 for g in loader.dataset[0])
    assert all(len(g.n_node) == 1 for g in loader.dataset[1])


# ---- Iteration / batching -------------------------------------------------

def test_loader_batches_all_inputs():
    n, bs = 10, 3
    graphs = _n_graphs(n)
    loader = gcnn.data.GraphLoader(graphs, batch_size=bs)
    batches = list(loader)
    assert len(batches) == math.ceil(n / bs)


@pytest.mark.parametrize("batch_mode", [gcnn.data.BatchMode.IMPLICIT, gcnn.data.BatchMode.EXPLICIT])
def test_loader_last_batch_is_reminder(batch_mode):
    n, bs = 19, 7
    graphs = _n_graphs(n)
    loader = gcnn.data.GraphLoader(graphs, batch_size=bs, batch_mode=batch_mode)
    batches = list(loader)

    num_batches = math.ceil(n / bs)
    assert len(batches) == num_batches

    last = batches[-1][0]
    if batch_mode is gcnn.data.BatchMode.IMPLICIT:
        # Real graph count on the last batch: 19 - 7 - 7 = 5 graphs of 2 nodes each
        expected_graphs = n - 2 * bs  # = 5
        assert int(np.asarray(last.n_node).sum()) == expected_graphs * 2
    else:  # EXPLICIT
        # The last batch is padded with dummy graphs, so all "graphs" are present
        # as (batch_size, 2) but the last entries are dummies.
        assert last.n_node.shape == (bs, 2)


def test_loader_iteration_is_repeatable():
    graphs = _n_graphs(5)
    loader = gcnn.data.GraphLoader(graphs, batch_size=2, shuffle=False)
    first = [b for b in loader]
    second = [b for b in loader]
    assert len(first) == len(second)


# ---- Shuffle ---------------------------------------------------------------

def test_loader_shuffles_ordering():
    # Give each graph a distinguishable global value so shuffling is observable.
    graphs = _graphs_with_globals(8)

    loader = gcnn.data.GraphLoader(graphs, batch_size=2, shuffle=True)
    all_g = jnp.concatenate([b[0].globals["g"] for b in loader])
    # Whatever the shuffle order, we must see every original graph exactly once
    assert sorted(np.asarray(all_g).tolist()) == list(range(8))


# ---- None targets ----------------------------------------------------------

def test_loader_allows_none_targets():
    graphs = _n_graphs(5)
    loader = gcnn.data.GraphLoader(graphs, None, batch_size=2)
    for batch in loader:
        assert len(batch) == 2
        assert isinstance(batch[0], jraph.GraphsTuple)
        assert batch[1] is None


def test_loader_mixed_none_and_graph_datasets():
    graphs = _n_graphs(4)
    loader = gcnn.data.GraphLoader(graphs, None, graphs, batch_size=2)
    assert len(loader) == 2
    for batch in loader:
        assert batch[1] is None
        assert isinstance(batch[0], jraph.GraphsTuple)
        assert isinstance(batch[2], jraph.GraphsTuple)


# ---- Sampler API -----------------------------------------------------------

def test_loader_with_new_sampler_preserves_params():
    graphs = _n_graphs(8)
    loader = gcnn.data.GraphLoader(graphs, batch_size=4, pad=True,
                                   batch_mode=gcnn.data.BatchMode.IMPLICIT)
    new_sampler = samplers.BatchSampler(
        samplers.SequentialSampler(8), batch_size=4, drop_last=True
    )
    new_loader = loader.with_new_sampler(new_sampler)
    assert new_loader.batch_size == 4
    assert new_loader.shuffle is loader.shuffle
    assert new_loader.sampler is new_sampler
    # The new loader must yield the same number of batches as the new sampler says
    assert len(new_loader) == len(new_sampler)


def test_loader_with_new_sampler_changes_iteration():
    """Two loaders on the same dataset with different samplers yield different
    number of batches (the new sampler is actually used)."""
    graphs = _n_graphs(8)
    loader = gcnn.data.GraphLoader(graphs, batch_size=4)

    # Replace with drop_last=True -> 8/4 == 2 batches (drop_last is True, 8 % 4 == 0, so 2)
    sampler_drop = samplers.BatchSampler(
        samplers.SequentialSampler(8), batch_size=4, drop_last=True
    )
    assert len(loader.with_new_sampler(sampler_drop)) == 2

    # Replace with batch_size=2 -> 4 batches
    sampler_smaller = samplers.BatchSampler(
        samplers.SequentialSampler(8), batch_size=2, drop_last=False
    )
    assert len(loader.with_new_sampler(sampler_smaller)) == 4


def test_loader_with_new_sampler_returns_new_instance():
    graphs = _n_graphs(4)
    loader = gcnn.data.GraphLoader(graphs, batch_size=2)
    new_sampler = samplers.BatchSampler(
        samplers.SequentialSampler(4), batch_size=2, drop_last=False
    )
    new_loader = loader.with_new_sampler(new_sampler)
    assert new_loader is not loader


# ---- Empty batch handling --------------------------------------------------

def test_empty_batch_yields_fully_masked_graph():
    """The distributed sampler may hand a rank a completely empty batch to
    keep the per-rank step count uniform. ``GraphLoader`` must turn that
    into a valid, entirely-masked graph.
    """
    batch_size = 4
    graphs = _n_graphs(2)

    inner = samplers.BatchSampler(
        samplers.SequentialSampler(len(graphs)), batch_size=batch_size, drop_last=False
    )
    empty_sampler = samplers.DistributedSampler(inner=inner, num_replicas=2, process_index=1)
    assert list(empty_sampler) == [
        [],
    ], "test setup assumes rank 1 yields one empty batch"

    for batch_mode in (gcnn.data.BatchMode.EXPLICIT, gcnn.data.BatchMode.IMPLICIT):
        loader = gcnn.data.GraphLoader(
            graphs,
            batch_size=batch_size,
            pad=True,
            batch_mode=batch_mode,
            sampler=empty_sampler.__class__(inner, num_replicas=2, process_index=1),
        )
        batches = list(loader)
        assert len(batches) == 1
        batch = batches[0][0]

        if batch_mode is gcnn.data.BatchMode.EXPLICIT:
            assert batch.n_node.shape == (batch_size, 2)
        else:
            assert batch.n_node.shape == (loader.padding.n_graphs,)

        # All real graphs are empty
        assert int(np.asarray(batch.n_node).sum()) == 0
        assert int(np.asarray(batch.n_edge).sum()) == 0

        # Every node, edge and graph is masked out
        node_mask = np.asarray(batch.nodes[keys.MASK])
        edge_mask = np.asarray(batch.edges[keys.MASK])
        graph_mask = np.asarray(batch.globals[keys.MASK])
        assert not node_mask.any()
        assert not edge_mask.any()
        assert not graph_mask.any()
