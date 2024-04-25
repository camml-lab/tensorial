# -*- coding: utf-8 -*-
import math

import jax.numpy as jnp
import jraph
import numpy as np
import utils

from tensorial.gcnn import datasets, keys


def test_generate_batches():
    dataset_size = 5
    batch_size = 2
    inputs = tuple(utils.random_spatial_graph(2, cutoff=5) for _ in range(dataset_size))
    batches = tuple(datasets.generate_batches(batch_size, inputs))

    padded = tuple(datasets.generated_padded_graphs(batches))
    num_nodes = sum(padded[0].inputs.n_node)
    num_edges = sum(padded[0].inputs.n_edge)
    num_graphs = len(padded[0].inputs.n_edge)
    for inputs, _targets in padded[1:]:
        assert sum(inputs.n_node) == num_nodes
        assert sum(inputs.n_edge) == num_edges
        assert len(inputs.n_node) == num_graphs

    # Check that the padding is as we expected
    for inputs, _targets in padded[:-1]:
        # There should be one graph added to the end for padding
        assert jraph.get_number_of_padding_with_graphs_graphs(inputs) == 1
        graph_mask = jraph.get_graph_padding_mask(inputs)
        assert jnp.all(graph_mask[:-1])
        assert graph_mask[-1].item() is False


def test_generate_batches_with_mask():
    dataset_size = 5
    batch_size = 2
    inputs = tuple(utils.random_spatial_graph(2, cutoff=3) for _ in range(dataset_size))
    batches = tuple(datasets.generate_batches(batch_size, inputs))

    padded = tuple(datasets.generated_padded_graphs(batches, add_mask=True))
    for batch_idx in (0, -1):
        batch = padded[batch_idx]
        assert jnp.all(batch.inputs.globals[keys.DEFAULT_PAD_MASK_FIELD] == jraph.get_graph_padding_mask(batch.inputs))
        assert jnp.all(batch.inputs.nodes[keys.DEFAULT_PAD_MASK_FIELD] == jraph.get_node_padding_mask(batch.inputs))
        assert jnp.all(batch.inputs.edges[keys.DEFAULT_PAD_MASK_FIELD] == jraph.get_edge_padding_mask(batch.inputs))


def test_create_batches():
    dataset_size = 19
    batch_size = 7
    num_batches = math.ceil(dataset_size / batch_size)
    inputs = tuple(utils.random_spatial_graph(2) for _ in range(dataset_size))

    batches = tuple(datasets.generate_batches(batch_size, inputs))
    assert len(batches) == num_batches
    for inputs, _outputs in batches[:num_batches - 1]:
        assert len(inputs.n_node) == batch_size

    # Check the last batch has the remainder
    assert len(batches[-1].inputs.n_node) == dataset_size - (num_batches - 1) * batch_size


def test_create_batches_from_graph_batch():
    dataset_size = 7
    batch_size = 2
    inputs = tuple(utils.random_spatial_graph(2) for _ in range(dataset_size))

    # Now create a batch from these graphs
    single_batch = jraph.batch(inputs)
    batches = tuple(datasets.generate_batches(batch_size, single_batch, outputs=single_batch))
    assert len(batches) == math.ceil(dataset_size / batch_size)
    assert isinstance(batches[0].inputs, jraph.GraphsTuple)
    assert isinstance(batches[0].targets, jraph.GraphsTuple)


def test_add_pading_mask():
    num_nodes = 3
    graph = utils.random_spatial_graph(num_nodes, cutoff=3)
    num_edges = graph.n_edge[0].item()
    mask_field = 'my_mask'

    padded = jraph.pad_with_graphs(graph, num_nodes + 1, num_edges + 1, 2)
    assert len(padded.n_node) == 2
    assert sum(padded.n_node) == num_nodes + 1
    assert sum(padded.n_edge) == num_edges + 1

    padded = datasets.add_padding_mask(padded, mask_field)
    # Nodes
    mask = np.ones(num_nodes + 1, dtype=bool)
    mask[-1] = False
    assert jnp.all(padded.nodes[mask_field] == mask)
    # Edges
    mask = np.ones(num_edges + 1, dtype=bool)
    mask[-1] = False
    assert jnp.all(padded.edges[mask_field] == mask)
    # Globals
    mask = np.array([True, False])
    assert jnp.all(padded.globals[mask_field] == mask)
