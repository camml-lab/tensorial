"""Tests for :class:`tensorial.gcnn.data.GraphBatcher` and related padding helpers."""

import math

import jax
import jax.numpy as jnp
import jraph
import numpy as np
import pytest

from tensorial import gcnn
from tensorial.gcnn import keys


def _unbatch_explicit(batched_graph: jraph.GraphsTuple) -> list[jraph.GraphsTuple]:
    """Unbatch an explicitly batched GraphsTuple into a list of individual
    GraphsTuples. Each graph keeps its original single-graph shape for
    ``n_node`` / ``n_edge``."""
    batch_size = jax.tree_util.tree_leaves(batched_graph.n_node)[0].shape[0]
    graphs = []
    for i in range(batch_size):
        node = jax.tree_util.tree_map(lambda x: x[i], batched_graph.nodes)
        edge = jax.tree_util.tree_map(lambda x: x[i], batched_graph.edges)
        glob = jax.tree_util.tree_map(lambda x: x[i], batched_graph.globals)
        g = jraph.GraphsTuple(
            nodes=node,
            edges=edge,
            senders=batched_graph.senders[i],
            receivers=batched_graph.receivers[i],
            n_node=batched_graph.n_node[i : i + 1],
            n_edge=batched_graph.n_edge[i : i + 1],
            globals=glob,
        )
        graphs.append(g)
    return graphs


# ---- Implicit batching -----------------------------------------------------

def test_generate_batches(rng_key):
    """Implicit batching produces batches of uniform padded size."""
    dataset_size = 5
    batch_size = 2
    inputs = tuple(gcnn.random.spatial_graph(rng_key, 2, cutoff=5) for _ in range(dataset_size))
    batches = tuple(gcnn.data.GraphBatcher(inputs, batch_size=batch_size, pad=True))

    num_nodes = sum(batches[0].n_node)
    num_edges = sum(batches[0].n_edge)
    num_graphs = len(batches[0].n_edge)
    for graph in batches[1:]:
        assert sum(graph.n_node) == num_nodes
        assert sum(graph.n_edge) == num_edges
        assert len(graph.n_node) == num_graphs

    # All but the last batch have exactly one padding graph appended
    for graph in batches[:-1]:
        assert jraph.get_number_of_padding_with_graphs_graphs(graph) == 1
        graph_mask = jraph.get_graph_padding_mask(graph)
        assert jnp.all(graph_mask[:-1])
        assert graph_mask[-1].item() is False


def test_create_batches(rng_key):
    """Without padding, each batch has exactly ``batch_size`` graphs except
    the last one, which holds the remainder."""
    dataset_size = 19
    batch_size = 7
    num_batches = math.ceil(dataset_size / batch_size)
    dset = tuple(gcnn.random.spatial_graph(rng_key, 2) for _ in range(dataset_size))

    batches = tuple(gcnn.data.GraphBatcher(dset, batch_size=batch_size))
    assert len(batches) == num_batches
    for batch in batches[: num_batches - 1]:
        assert len(batch.n_node) == batch_size

    # Last batch holds the remainder graphs
    assert len(batches[-1].n_node) == dataset_size - (num_batches - 1) * batch_size


def test_generate_batches_with_mask(rng_key):
    """When ``add_mask=True`` the mask stored in the graph matches the padding
    mask that jraph would compute."""
    dataset_size = 5
    batch_size = 2
    inputs = tuple(gcnn.random.spatial_graph(rng_key, 2, cutoff=3) for _ in range(dataset_size))
    batches = tuple(
        gcnn.data.GraphBatcher(inputs, batch_size=batch_size, pad=True, add_mask=True)
    )

    # Check the first and last batch (the last has fewer real graphs)
    for batch_idx in (0, -1):
        batch = batches[batch_idx]
        assert jnp.all(batch.globals[keys.MASK] == jraph.get_graph_padding_mask(batch))
        assert jnp.all(batch.nodes[keys.MASK] == jraph.get_node_padding_mask(batch))
        assert jnp.all(batch.edges[keys.MASK] == jraph.get_edge_padding_mask(batch))


# ---- Explicit batching -----------------------------------------------------

@pytest.mark.parametrize("drop_last", [True, False])
def test_generate_batches_explicit(rng_key, drop_last: bool):
    dataset_size = 5
    batch_size = 2

    inputs = []
    for _ in range(dataset_size):
        rng_key, _ = jax.random.split(rng_key)
        inputs.append(gcnn.random.spatial_graph(rng_key, 2, cutoff=5))

    batcher = gcnn.data.GraphBatcher(
        inputs,
        batch_size=batch_size,
        pad=True,
        mode=gcnn.data.BatchMode.EXPLICIT,
        drop_last=drop_last,
    )
    batches = tuple(batcher)
    assert len(batcher) == 5 // 2 if drop_last else 5 // 2 + 1

    input_idx = 0
    for batch in batches:
        unbatched = _unbatch_explicit(batch)
        for entry in unbatched:
            if input_idx == len(inputs) - 1:
                assert drop_last is False
                break

            input = inputs[input_idx]
            padded = gcnn.data.pad_with_graphs(input, *batcher.padding)
            for a, b in zip(jax.tree.flatten(padded)[0], jax.tree.flatten(entry)[0]):
                assert jnp.all(a == b)
            input_idx += 1

    # All but the last batch have exactly one padding graph appended
    for batch in batches[:-1]:
        assert jraph.get_number_of_padding_with_graphs_graphs(batch) == 1
        graph_mask = jraph.get_graph_padding_mask(batch)
        assert jnp.all(graph_mask[:-1])
        assert graph_mask[-1].item() is False


# ---- Padding helpers -------------------------------------------------------

def test_add_padding_mask(cube_graph: jraph.GraphsTuple):
    """``add_padding_mask`` should produce a ``keys.MASK`` whose first
    ``n_node`` entries are True and the padding nodes are False, and respect
    the ``overwrite`` flag."""
    mask = np.zeros(dtype=bool, shape=(len(cube_graph.nodes[keys.POSITIONS]),))
    mask[0] = True
    cube_graph.nodes[keys.MASK] = mask
    padded = jraph.pad_with_graphs(
        cube_graph, n_node=cube_graph.n_node.item() + 1, n_edge=cube_graph.n_edge.item() + 1
    )
    padded = gcnn.data.add_padding_mask(padded)

    assert np.all(padded.nodes[keys.MASK][:-1] == mask)
    assert padded.nodes[keys.MASK][-1].item() is False

    # With ``overwrite=True`` the mask is recomputed from scratch.
    padded = gcnn.data.add_padding_mask(padded, overwrite=True)
    assert np.all(padded.nodes[keys.MASK] == np.array([True] * cube_graph.n_node.item() + [False]))


def test_pad_with_graphs(cube_graph_gcnn: jraph.GraphsTuple):
    """``pad_with_graphs`` should append ``extra`` graphs while also writing
    padding masks into nodes, edges and globals."""
    extra_nodes = 1
    extra_edges = 0
    extra_graphs = 1
    padded = gcnn.data.pad_with_graphs(
        cube_graph_gcnn,
        cube_graph_gcnn.n_node[0] + extra_nodes,
        cube_graph_gcnn.n_edge[0] + extra_edges,
        len(cube_graph_gcnn.n_node) + extra_graphs,
    )

    # The mask is present in all graph attribute groups
    for key in ("nodes", "edges", "globals"):
        assert keys.MASK in padded._asdict()[key]

    assert padded.n_node.tolist() == [cube_graph_gcnn.n_node[0], extra_nodes]
    assert padded.n_edge.tolist() == [cube_graph_gcnn.n_edge[0], extra_edges]
    assert len(padded.n_node) == len(cube_graph_gcnn.n_node) + extra_graphs


def test_max_padding():
    """``max_padding`` should pick the element-wise maximum across paddings."""
    a = gcnn.data.GraphPadding(n_nodes=2, n_edges=3, n_graphs=2)
    b = gcnn.data.GraphPadding(n_nodes=5, n_edges=1, n_graphs=4)
    c = gcnn.data.GraphPadding(n_nodes=1, n_edges=7, n_graphs=3)

    # Elementwise max of {a, b} = n_nodes:5 (b), n_edges:3 (a), n_graphs:4 (b)
    assert gcnn.data.max_padding(a, b) == gcnn.data.GraphPadding(5, 3, 4)
    # Elementwise max of {a, b, c} = n_nodes:5 (b), n_edges:7 (c), n_graphs:4 (b)
    assert gcnn.data.max_padding(a, b, c) == gcnn.data.GraphPadding(5, 7, 4)


# ---- Error cases -----------------------------------------------------------

def test_batcher_rejects_batched_sequence_input():
    """A *sequence* of batched graphs is invalid: only a single batched
    ``GraphsTuple`` (which is unbatched) or a sequence of individual graphs
    is accepted."""
    from tensorial.gcnn.data import GraphBatcher

    g = jraph.GraphsTuple(
        n_node=np.array([2]),
        n_edge=np.array([3]),
        nodes={"f": np.ones((2, 1), dtype=np.float32)},
        edges={"f": np.ones((3, 1), dtype=np.float32)},
        globals={"g": np.ones((1,), dtype=np.float32)},
        senders=np.zeros(3, dtype=np.int32),
        receivers=np.zeros(3, dtype=np.int32),
    )
    batched = jraph.batch_np([g, g])  # n_node has len 2
    with pytest.raises(ValueError, match="individual graphs"):
        GraphBatcher([batched], batch_size=2)


def test_fetch_rejects_over_batch_size():
    """``fetch`` must reject index lists longer than the batch size."""
    from tensorial.gcnn.data import GraphBatcher

    g = jraph.GraphsTuple(
        n_node=np.array([2]),
        n_edge=np.array([3]),
        nodes={"f": np.ones((2, 1), dtype=np.float32)},
        edges={"f": np.ones((3, 1), dtype=np.float32)},
        globals={"g": np.ones((1,), dtype=np.float32)},
        senders=np.zeros(3, dtype=np.int32),
        receivers=np.zeros(3, dtype=np.int32),
    )
    batcher = GraphBatcher([g, g], batch_size=2)
    with pytest.raises(ValueError, match="batch size"):
        batcher.fetch([0, 1, 1])
