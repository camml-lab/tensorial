from typing import Final

import e3nn_jax as e3j
import jax
from jax import random
import jax.numpy as jnp
import jraph

import tensorial
from tensorial import gcnn
import tensorial.tensors


def test_edgewise_linear(rng_key):
    in_field: Final[str] = "in"
    out_field: Final[str] = "out"
    in_irreps = e3j.Irreps("2x0e+2x1o")
    out_irreps = e3j.Irreps("1o")
    n_edges = 10

    # Create some random node attributes
    edge_attrs = e3j.IrrepsArray(in_irreps, random.uniform(rng_key, (n_edges, in_irreps.dim)))

    graph = jraph.GraphsTuple(
        nodes=None,
        senders=None,
        receivers=None,
        edges={in_field: edge_attrs},
        globals=None,
        n_node=jnp.array([0]),
        n_edge=jnp.array([n_edges]),
    )

    linear = gcnn.EdgewiseLinear(irreps_out=out_irreps, field=in_field, out_field=out_field)
    params = linear.init(rng_key, graph)
    out_graph = linear.apply(params, graph)

    assert out_field in out_graph.edges
    assert isinstance(out_graph.edges[out_field], e3j.IrrepsArray)
    assert out_graph.edges[out_field].irreps == out_irreps
    assert out_graph.edges[out_field].shape[0] == n_edges


def test_edgewise_encoding(rng_key):
    in_field: Final[str] = "in"
    out_field: Final[str] = "out"
    n_edges = 10
    num_elements: Final[int] = 2

    # Let's use a one-hot for testing
    one_hot = tensorial.tensors.OneHot(num_elements)
    edge_attrs = random.randint(rng_key, (n_edges,), 0, num_elements)

    graph = jraph.GraphsTuple(
        nodes=None,
        senders=None,
        receivers=None,
        edges={in_field: edge_attrs},
        globals=None,
        n_node=jnp.array([0]),
        n_edge=jnp.array([n_edges]),
    )

    encoding = gcnn.EdgewiseEmbedding({in_field: one_hot}, out_field=out_field)
    out_graph = encoding(graph)
    assert out_field in out_graph.edges
    assert isinstance(out_graph.edges[out_field], e3j.IrrepsArray)
    assert out_graph.edges[out_field].irreps == one_hot.irreps
    assert out_graph.edges[out_field].shape[0] == n_edges


def test_edgewise_decoding(rng_key):
    in_field: Final[str] = "in"
    out_field: Final[str] = "out"
    n_edges = 10

    # Let's use a one-hot for testing
    cart = tensorial.CartesianTensor("ij=ji", i="1e")
    edge_attrs = e3j.IrrepsArray(cart.irreps, random.uniform(rng_key, (n_edges, cart.irreps.dim)))

    graph = jraph.GraphsTuple(
        nodes=None,
        senders=None,
        receivers=None,
        edges={in_field: edge_attrs},
        globals=None,
        n_node=jnp.array([0]),
        n_edge=jnp.array([n_edges]),
    )

    decoding = gcnn.EdgewiseDecoding(
        attrs={out_field: cart},
        in_field=in_field,
    )

    out_graph = decoding(graph)
    assert out_field in out_graph.edges
    assert isinstance(out_graph.edges[out_field], jax.Array)
    assert out_graph.edges[out_field].shape == (n_edges, 3, 3)
