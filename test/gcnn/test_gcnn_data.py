import math

import jax
import jax.numpy as jnp
import jraph
import numpy as np
import pytest

from tensorial import gcnn
from tensorial.gcnn import data, keys


def unbatch_explicit(batched_graph: jraph.GraphsTuple) -> list[jraph.GraphsTuple]:
    """Unbatch an explicitly batched GraphsTuple into a list of GraphsTuples
    where nodes, edges, and globals are PyTrees (e.g. dictionaries of arrays)."""
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
            n_node=batched_graph.n_node[i : i + 1],  # this should remain a [1] shape
            n_edge=batched_graph.n_edge[i : i + 1],
            globals=glob,
        )
        graphs.append(g)

    return graphs


def test_generate_batches(rng_key):
    dataset_size = 5
    batch_size = 2
    inputs = tuple(gcnn.random.spatial_graph(rng_key, 2, cutoff=5) for _ in range(dataset_size))
    batches = tuple(data.GraphBatcher(inputs, batch_size=batch_size, pad=True))

    num_nodes = sum(batches[0].n_node)
    num_edges = sum(batches[0].n_edge)
    num_graphs = len(batches[0].n_edge)
    for graph in batches[1:]:
        assert sum(graph.n_node) == num_nodes
        assert sum(graph.n_edge) == num_edges
        assert len(graph.n_node) == num_graphs

    # Check that the padding is as we expected
    for graph in batches[:-1]:
        # There should be one graph added to the end for padding
        assert jraph.get_number_of_padding_with_graphs_graphs(graph) == 1
        graph_mask = jraph.get_graph_padding_mask(graph)
        assert jnp.all(graph_mask[:-1])
        assert graph_mask[-1].item() is False


@pytest.mark.parametrize("drop_last", [True, False])
def test_generate_batches_explicit(rng_key, drop_last: bool):
    dataset_size = 5
    batch_size = 2

    # Generate the random graphs
    inputs = []
    for _ in range(dataset_size):
        rng_key, graph_key = jax.random.split(rng_key)
        inputs.append(gcnn.random.spatial_graph(rng_key, 2, cutoff=5))

    batcher = data.GraphBatcher(
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
        unbatched = unbatch_explicit(batch)

        for entry in unbatched:
            if input_idx == len(inputs) - 1:
                assert drop_last is False
                break

            input = inputs[input_idx]
            padded = gcnn.data.pad_with_graphs(input, *batcher.padding)
            for a, b in zip(jax.tree.flatten(padded)[0], jax.tree.flatten(entry)[0]):
                assert jnp.all(a == b)

            input_idx += 1

    # Check that the padding is as we expected
    for batch in batches[:-1]:
        # There should be one graph added to the end for padding
        assert jraph.get_number_of_padding_with_graphs_graphs(batch) == 1
        graph_mask = jraph.get_graph_padding_mask(batch)
        assert jnp.all(graph_mask[:-1])
        assert graph_mask[-1].item() is False


def test_generate_batches_with_mask(rng_key):
    dataset_size = 5
    batch_size = 2
    inputs = tuple(gcnn.random.spatial_graph(rng_key, 2, cutoff=3) for _ in range(dataset_size))
    batches = tuple(data.GraphBatcher(inputs, batch_size=batch_size, pad=True, add_mask=True))

    # Check the first and last batch (which only has one graph)
    for batch_idx in (0, -1):
        batch = batches[batch_idx]
        assert jnp.all(batch.globals[keys.MASK] == jraph.get_graph_padding_mask(batch))
        assert jnp.all(batch.nodes[keys.MASK] == jraph.get_node_padding_mask(batch))
        assert jnp.all(batch.edges[keys.MASK] == jraph.get_edge_padding_mask(batch))


def test_create_batches(rng_key):
    dataset_size = 19
    batch_size = 7
    num_batches = math.ceil(dataset_size / batch_size)
    dset = tuple(gcnn.random.spatial_graph(rng_key, 2) for _ in range(dataset_size))

    batches = tuple(data.GraphBatcher(dset, batch_size=batch_size))
    assert len(batches) == num_batches
    for batch in batches[: num_batches - 1]:
        assert len(batch.n_node) == batch_size

    # Check the last batch has the remainder
    assert len(batches[-1].n_node) == dataset_size - (num_batches - 1) * batch_size


def test_graph_loader(rng_key):
    dataset_size = 19
    batch_size = 7
    num_batches = math.ceil(dataset_size / batch_size)
    # Check that we can have inputs and outputs with different numbers of nodes
    inputs = tuple(gcnn.random.spatial_graph(rng_key, 2) for _ in range(dataset_size))
    targets = tuple(gcnn.random.spatial_graph(rng_key, 3) for _ in range(dataset_size))

    batches = tuple(data.GraphLoader(inputs, targets, batch_size=batch_size))
    assert len(batches) == num_batches
    for batch_inputs, batch_targets in batches[: num_batches - 1]:
        assert len(batch_inputs.n_node) == batch_size
        assert len(batch_targets.n_node) == batch_size

    # Check the last batch has the remainder
    assert len(batches[-1][0].n_node) == dataset_size - (num_batches - 1) * batch_size

    # Now test that we can pass None for the targets (this is often done when the input graph itself
    # contains the labels)
    batches = tuple(data.GraphLoader(inputs, None, batch_size=batch_size))
    for batch_inputs, batch_targets in batches:
        assert isinstance(batch_inputs, jraph.GraphsTuple)
        assert batch_targets is None


def test_add_padding_mask(cube_graph: jraph.GraphsTuple):
    mask = np.zeros(dtype=bool, shape=(len(cube_graph.nodes[keys.POSITIONS]),))
    mask[0] = True
    cube_graph.nodes[keys.MASK] = mask
    padded = jraph.pad_with_graphs(
        cube_graph, n_node=cube_graph.n_node.item() + 1, n_edge=cube_graph.n_edge.item() + 1
    )
    padded = data.add_padding_mask(padded)

    assert np.all(padded.nodes[keys.MASK][:-1] == mask)
    assert padded.nodes[keys.MASK][-1].item() is False

    # Test overwrite
    padded = data.add_padding_mask(padded, overwrite=True)
    assert np.all(padded.nodes[keys.MASK] == np.array([*[True] * cube_graph.n_node.item(), False]))


def test_nequip_interaction_block_with_padding(cube_graph_gcnn: jraph.GraphsTuple, rng_key):
    extra_nodes = 1
    extra_edges = 0
    extra_graphs = 1
    padded = gcnn.data.pad_with_graphs(
        cube_graph_gcnn,
        cube_graph_gcnn.n_node[0] + extra_nodes,
        cube_graph_gcnn.n_edge[0] + extra_edges,
        len(cube_graph_gcnn.n_node) + extra_graphs,
    )

    # Check the mask has been added to all graph attributes
    for key in ("nodes", "edges", "globals"):
        assert "mask" in padded._asdict()[key]

    assert padded.n_node.tolist() == [cube_graph_gcnn.n_node[0], extra_nodes]
    assert padded.n_edge.tolist() == [cube_graph_gcnn.n_edge[0], extra_edges]
    assert len(padded.n_node) == len(cube_graph_gcnn.n_node) + extra_graphs
