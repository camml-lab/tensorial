# -*- coding: utf-8 -*-
import jax.numpy as jnp
import jraph

from tensorial.gcnn import _modules as modules


def test_rescale(rng_key):
    # Define a three node graph, each node has an integer as its feature.
    vals = jnp.array([[0.0], [1.0], [2.0]])
    shift = 12345.678
    scale = 5.76

    node_features = {'vals': vals}

    # We will construct a graph for which there is a directed edge between each node
    # and its successor. We define this with `senders` (source nodes) and `receivers`
    # (destination nodes).
    senders = jnp.array([0, 1, 2])
    receivers = jnp.array([1, 2, 0])

    # You can optionally add edge attributes.
    edges = jnp.array([[5.0], [6.0], [7.0]])

    # We then save the number of nodes and the number of edges.
    # This information is used to make running GNNs over multiple graphs
    # in a GraphsTuple possible.
    n_node = jnp.array([3])
    n_edge = jnp.array([3])

    # Optionally you can add `global` information, such as a graph label.
    global_context = jnp.array([[1]])
    graph = jraph.GraphsTuple(
        nodes=node_features,
        senders=senders,
        receivers=receivers,
        edges=edges,
        n_node=n_node,
        n_edge=n_edge,
        globals=global_context,
    )

    rescale = modules.Rescale(shift_fields='nodes.vals', shift=shift)
    params = rescale.init(rng_key, graph)
    out = rescale.apply(params, graph)
    assert jnp.all(out.nodes['vals'] == vals + shift)

    rescale = modules.Rescale(scale_fields='nodes.vals', scale=scale)
    params = rescale.init(rng_key, graph)
    out = rescale.apply(params, graph)
    assert jnp.all(out.nodes['vals'] == vals * scale)

    rescale = modules.Rescale(scale_fields='nodes.vals', shift_fields='nodes.vals', scale=scale, shift=shift)
    params = rescale.init(rng_key, graph)
    out = rescale.apply(params, graph)
    assert jnp.all(out.nodes['vals'] == vals * scale + shift)
