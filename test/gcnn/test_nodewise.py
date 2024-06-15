# -*- coding: utf-8 -*-
from typing import Final

import e3nn_jax as e3j
import jax
from jax import random
import jax.numpy as jnp
import jraph
import pytest

import tensorial
from tensorial import gcnn
import tensorial.tensors


def test_nodewise_linear(rng_key):
    in_field: Final[str] = "in"
    out_field: Final[str] = "out"
    in_irreps = e3j.Irreps("2x0e+2x1o")
    out_irreps = e3j.Irreps("1o")
    n_nodes = 5

    # Create some random node attributes
    node_attrs = e3j.IrrepsArray(in_irreps, random.uniform(rng_key, (n_nodes, in_irreps.dim)))

    graph = jraph.GraphsTuple(
        nodes={in_field: node_attrs},
        senders=None,
        receivers=None,
        edges=None,
        globals=None,
        n_node=jnp.array([n_nodes]),
        n_edge=jnp.array([0]),
    )

    linear = gcnn.NodewiseLinear(irreps_out=out_irreps, field=in_field, out_field=out_field)
    params = linear.init(rng_key, graph)
    out_graph = linear.apply(params, graph)

    assert out_field in out_graph.nodes
    assert isinstance(out_graph.nodes[out_field], e3j.IrrepsArray)
    assert out_graph.nodes[out_field].irreps == out_irreps
    assert out_graph.nodes[out_field].shape[0] == n_nodes


def test_nodewise_encoding(rng_key):
    in_field: Final[str] = "in"
    out_field: Final[str] = "out"
    n_nodes = 5

    # Let's use a one-hot for testing
    one_hot = tensorial.tensors.OneHot(2)
    node_attrs = random.uniform(rng_key, (n_nodes,))

    graph = jraph.GraphsTuple(
        nodes={in_field: node_attrs},
        senders=None,
        receivers=None,
        edges=None,
        globals=None,
        n_node=jnp.array([n_nodes]),
        n_edge=jnp.array([0]),
    )

    encoding = gcnn.NodewiseEncoding({in_field: one_hot}, out_field=out_field)
    out_graph = encoding(graph)
    assert out_field in out_graph.nodes
    assert isinstance(out_graph.nodes[out_field], e3j.IrrepsArray)
    assert out_graph.nodes[out_field].irreps == one_hot.irreps
    assert out_graph.nodes[out_field].shape[0] == n_nodes


@pytest.mark.skip(reason="Currently there's a bug in flax that converts Irreps to a tuple")
def test_nodewise_encoding_multiple(rng_key):
    ont_hot_key: Final[str] = "in"
    scalar_key = "scalar"
    out_field: Final[str] = "out"
    n_nodes = 5

    keys = jax.random.split(rng_key, num=2)

    # Let's use a one-hot for testing
    one_hot = tensorial.tensors.OneHot(2)
    one_hots = random.uniform(keys[0], (n_nodes,))
    scalars = random.uniform(keys[1], (n_nodes, 1))
    scalar_irreps = e3j.Irreps("0e")

    graph = jraph.GraphsTuple(
        nodes={ont_hot_key: one_hots, scalar_key: scalars},
        senders=None,
        receivers=None,
        edges=None,
        globals=None,
        n_node=jnp.array([n_nodes]),
        n_edge=jnp.array([0]),
    )

    encoding = gcnn.NodewiseEncoding(
        {ont_hot_key: one_hot, scalar_key: scalar_irreps}, out_field=out_field
    )
    _ = encoding.init(rng_key, graph)
    out_graph = encoding(graph)
    assert out_field in out_graph.nodes
    assert isinstance(out_graph.nodes[out_field], e3j.IrrepsArray)
    assert out_graph.nodes[out_field].irreps == one_hot.irreps + scalar_irreps
    assert out_graph.nodes[out_field].shape[0] == n_nodes


def test_nodewise_encoding_compilation(rng_key):
    in_field: Final[str] = "in"
    out_field: Final[str] = "out"
    n_nodes = 5

    # Let's use a one-hot for testing
    one_hot = tensorial.tensors.OneHot(2)
    node_attrs = random.uniform(rng_key, (n_nodes,))

    graph = jraph.GraphsTuple(
        nodes={in_field: node_attrs},
        senders=None,
        receivers=None,
        edges=None,
        globals=None,
        n_node=jnp.array([n_nodes]),
        n_edge=jnp.array([0]),
    )

    encoding = gcnn.NodewiseEncoding({in_field: one_hot}, out_field=out_field)
    params = encoding.init(rng_key, graph)

    jitted = jax.jit(encoding.apply)

    updated = jitted(params, graph)  # pylint: disable=not-callable
    updated = jitted(params, updated)  # pylint: disable=not-callable
    updated = jitted(params, updated)  # pylint: disable=not-callable
    updated = jitted(params, updated)  # pylint: disable=not-callable


def test_nodewise_decoding(rng_key):
    in_field: Final[str] = "in"
    out_field: Final[str] = "out"
    n_nodes = 5

    # Let's use a one-hot for testing
    cart = tensorial.CartesianTensor("ij=ji", i="1e")
    node_attrs = e3j.IrrepsArray(cart.irreps, random.uniform(rng_key, (n_nodes, cart.irreps.dim)))

    graph = jraph.GraphsTuple(
        nodes={in_field: node_attrs},
        senders=None,
        receivers=None,
        edges=None,
        globals=None,
        n_node=jnp.array([n_nodes]),
        n_edge=jnp.array([0]),
    )

    decoding = gcnn.NodewiseDecoding(
        attrs={out_field: cart},
        in_field=in_field,
    )

    out_graph = decoding(graph)
    assert out_field in out_graph.nodes
    assert isinstance(out_graph.nodes[out_field], jax.Array)
    assert out_graph.nodes[out_field].shape == (n_nodes, 3, 3)


@pytest.mark.parametrize("jit", (True, False))
def test_nodewise_reduce(jit, rng_key):
    in_field: Final[str] = "in"
    out_field: Final[str] = "out"
    n_nodes = 5

    node_attrs = random.uniform(rng_key, (n_nodes,))

    graph = jraph.GraphsTuple(
        nodes={in_field: node_attrs},
        senders=None,
        receivers=None,
        edges=None,
        globals=None,
        n_node=jnp.array([n_nodes]),
        n_edge=jnp.array([0]),
    )

    decoding = gcnn.NodewiseReduce(field=in_field, out_field=out_field)
    params = decoding.init(random.PRNGKey(0), graph)
    apply = decoding.apply if jit is False else jax.jit(decoding.apply)
    out_graph = apply(params, graph)

    assert out_field in out_graph.globals
    assert isinstance(out_graph.globals[out_field], jax.Array)
    assert out_graph.globals[out_field].shape == (1,)
    assert out_graph.globals[out_field] == jnp.sum(node_attrs)
