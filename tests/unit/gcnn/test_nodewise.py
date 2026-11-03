from typing import Final

import e3nn_jax as e3j
import jax
from jax import random
import jax.numpy as jnp
import jraph
import numpy as np
import pytest

import tensorial
from tensorial import gcnn
from tensorial.gcnn import keys
import tensorial.tensors


def test_nodewise_linear(cube_graph, rng_key):
    in_field: Final[str] = "in"
    out_field: Final[str] = "out"
    in_irreps = e3j.Irreps("2x0e+2x1o")
    out_irreps = e3j.Irreps("1o")
    n_nodes = cube_graph.n_node[0]

    # Create some random node attributes
    node_attrs = e3j.IrrepsArray(in_irreps, random.uniform(rng_key, (n_nodes, in_irreps.dim)))
    cube_graph.nodes[in_field] = node_attrs

    linear = gcnn.NodewiseLinear(irreps_out=out_irreps, field=in_field, out_field=out_field)
    params = linear.init(rng_key, cube_graph)
    out_graph = linear.apply(params, cube_graph)

    assert out_field in out_graph.nodes
    assert isinstance(out_graph.nodes[out_field], e3j.IrrepsArray)
    assert out_graph.nodes[out_field].irreps == out_irreps
    assert out_graph.nodes[out_field].shape[0] == n_nodes

    EXTRA_NODES = 1
    EXTRA_EDGES = 0
    EXTRA_GRAPHS = 1
    padded = gcnn.data.pad_with_graphs(
        cube_graph,
        n_node=cube_graph.n_node[0] + EXTRA_NODES,
        n_edge=cube_graph.n_edge[0] + EXTRA_EDGES,
        n_graph=1 + EXTRA_GRAPHS,
    )
    # Add some random data to the padded nodes in case the nodewise linear does something with them
    padded.nodes[in_field].array[n_nodes:] = random.uniform(rng_key, (1, in_irreps.dim))
    out_padded = linear.apply(params, padded)
    # Now unbatch back to extract the original graph
    out = jraph.unbatch(out_padded)[0]

    assert jnp.allclose(out_graph.nodes[out_field].array, out.nodes[out_field].array)


def test_nodewise_linear_with_mask(rng_key):
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
    num_types: Final[int] = 2

    # Let's use a one-hot for testing
    one_hot = tensorial.tensors.OneHot(num_types)
    node_attrs = random.randint(rng_key, (n_nodes, 1), minval=0, maxval=num_types)

    graph = jraph.GraphsTuple(
        nodes={in_field: node_attrs},
        senders=None,
        receivers=None,
        edges=None,
        globals=None,
        n_node=jnp.array([n_nodes]),
        n_edge=jnp.array([0]),
    )

    encoding = gcnn.NodewiseEmbedding({in_field: one_hot}, out_field=out_field)
    out_graph = encoding(graph)
    assert out_field in out_graph.nodes
    assert isinstance(out_graph.nodes[out_field], e3j.IrrepsArray)
    assert out_graph.nodes[out_field].irreps == one_hot.irreps
    assert out_graph.nodes[out_field].shape[0] == n_nodes


def test_nodewise_encoding_from_global(rng_key):
    in_field: Final[str] = "in"
    out_field: Final[str] = "out"
    n_nodes = 5
    num_types: Final[int] = 2

    # Let's use a one-hot for testing
    one_hot = tensorial.tensors.OneHot(num_types)
    global_attr = random.randint(rng_key, (1, 1), minval=0, maxval=num_types)

    graph = jraph.GraphsTuple(
        nodes={gcnn.keys.POSITIONS: np.random.rand(n_nodes, 3)},
        senders=None,
        receivers=None,
        edges=None,
        globals={in_field: global_attr},
        n_node=jnp.array([n_nodes]),
        n_edge=jnp.array([0]),
    )

    encoding = gcnn.NodewiseEmbedding({f"globals.{in_field}": one_hot}, out_field=out_field)
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
    num_types: Final[int] = 2

    keys = jax.random.split(rng_key, num=2)

    # Let's use a one-hot for testing
    one_hot = tensorial.tensors.OneHot(num_types)
    one_hots = random.randint(keys[0], (n_nodes, 1), minval=0, maxval=num_types)
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

    encoding = gcnn.NodewiseEmbedding(
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
    num_types: Final[int] = 2

    # Let's use a one-hot for testing
    one_hot = tensorial.tensors.OneHot(num_types)
    node_attrs = random.randint(rng_key, (n_nodes, 1), minval=0, maxval=num_types)

    graph = jraph.GraphsTuple(
        nodes={in_field: node_attrs},
        senders=None,
        receivers=None,
        edges=None,
        globals=None,
        n_node=jnp.array([n_nodes]),
        n_edge=jnp.array([0]),
    )

    encoding = gcnn.NodewiseEmbedding({in_field: one_hot}, out_field=out_field)
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


def _graph_with_attrs(pos, n_node, attrs_irreps="1x0e", mask=None, rng=None) -> jraph.GraphsTuple:
    pos = jnp.asarray(pos)
    n_nodes = pos.shape[0]
    rng = rng if rng is not None else random.PRNGKey(0)
    irreps = e3j.Irreps(attrs_irreps)
    attrs = e3j.IrrepsArray(irreps, random.uniform(rng, (n_nodes, irreps.dim)))

    nodes = {
        keys.POSITIONS: pos,
        keys.ATTRIBUTES: attrs,
        keys.MASK: jnp.ones(n_nodes, dtype=bool) if mask is None else jnp.asarray(mask),
        # A unit scalar leaves the spatial features unchanged by the time tensor product.
        "time_embedding": jnp.ones((n_nodes, 1)),
    }

    return jraph.GraphsTuple(
        nodes=nodes,
        senders=None,
        receivers=None,
        edges=None,
        globals=None,
        n_node=jnp.asarray(n_node),
        n_edge=jnp.array([0] * len(n_node)),
    )


def _long_range_features(layer, graph):
    input_dim = graph.nodes[layer.field].irreps.dim
    output = layer.apply({}, jax.tree_util.tree_map(lambda x: x, graph))
    return output.nodes[layer.field][..., input_dim:]


def test_all_pairs_long_range_encoding_irreps():
    graph = _graph_with_attrs(np.zeros((3, 3)), [3], attrs_irreps="2x0e")
    long_range = _long_range_features(gcnn.AllPairsLongRangeEncoding(l_max=2), graph)

    assert long_range.irreps == e3j.Irreps("0e + 1o + 2e")
    assert long_range.shape == (3, 9)


def test_all_pairs_long_range_encoding_pair_direction_and_no_self():
    separation = 2.5
    graph = _graph_with_attrs([[0.0, 0.0, 0.0], [separation, 0.0, 0.0]], [2])
    long_range = _long_range_features(gcnn.AllPairsLongRangeEncoding(l_max=1), graph)

    expected_positive = e3j.spherical_harmonics(
        "1o", jnp.array([1.0, 0.0, 0.0]), True, normalization="integral"
    ).array
    expected_negative = e3j.spherical_harmonics(
        "1o", jnp.array([-1.0, 0.0, 0.0]), True, normalization="integral"
    ).array
    # Centre j=0 receives from i=1, so x_j - x_i points along negative x.
    assert jnp.allclose(long_range.array[0, 1:], separation * expected_negative)
    assert jnp.allclose(long_range.array[1, 1:], separation * expected_positive)

    single = _graph_with_attrs([[3.0, -2.0, 1.0]], [1])
    single_features = _long_range_features(gcnn.AllPairsLongRangeEncoding(l_max=2), single)
    assert jnp.allclose(single_features.array, 0.0)
    assert not jnp.any(jnp.isnan(single_features.array))


def test_all_pairs_long_range_encoding_is_raw_sum():
    one_source = _graph_with_attrs([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [2])
    two_sources = _graph_with_attrs([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [3])
    layer = gcnn.AllPairsLongRangeEncoding(l_max=2, normalise=False)
    one = _long_range_features(layer, one_source)
    two = _long_range_features(layer, two_sources)

    assert jnp.allclose(two.array[0], 2.0 * one.array[0])


def test_all_pairs_long_range_encoding_radial_scaling():
    scale = 3.0
    graph = _graph_with_attrs([[0.0, 0.0, 0.0], [1.0, 2.0, -1.0]], [2])
    scaled_graph = _graph_with_attrs([[0.0, 0.0, 0.0], [scale, 2 * scale, -scale]], [2])
    layer = gcnn.AllPairsLongRangeEncoding(l_max=2)
    features = _long_range_features(layer, graph)
    scaled = _long_range_features(layer, scaled_graph)

    offsets = (0, 1, 4, 9)
    for l in range(3):  # noqa: E741
        block = slice(offsets[l], offsets[l + 1])
        assert jnp.allclose(scaled.array[:, block], scale**l * features.array[:, block])


def test_all_pairs_long_range_encoding_translation_invariance():
    pos = np.array([[0.2, 1.0, -0.5], [2.0, -1.0, 0.3], [-0.4, 0.5, 1.2]])
    translated = pos + np.array([10.0, -7.0, 3.5])
    layer = gcnn.AllPairsLongRangeEncoding(l_max=2)
    original = _long_range_features(layer, _graph_with_attrs(pos, [3]))
    shifted = _long_range_features(layer, _graph_with_attrs(translated, [3]))

    assert jnp.allclose(original.array, shifted.array, atol=1e-6)


def test_all_pairs_long_range_encoding_rotation_equivariance(rng_key):
    pos = e3j.IrrepsArray("1o", jnp.array([[0.2, 1.0, -0.5], [2.0, -1.0, 0.3], [-0.4, 0.5, 1.2]]))
    layer = gcnn.AllPairsLongRangeEncoding(l_max=2)

    def encode(positions):
        graph = _graph_with_attrs(positions.array, [3])
        return _long_range_features(layer, graph)

    e3j.utils.assert_equivariant(encode, rng_key, pos)


def test_all_pairs_long_range_encoding_batched_graph_isolation():
    first_pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, 0.0]])
    second_pos = np.array([[10.0, -4.0, 2.0], [11.0, -3.5, 2.0]])
    layer = gcnn.AllPairsLongRangeEncoding(l_max=2)
    batched = _long_range_features(
        layer, _graph_with_attrs(np.concatenate([first_pos, second_pos]), [2, 2])
    )
    first = _long_range_features(layer, _graph_with_attrs(first_pos, [2]))
    second = _long_range_features(layer, _graph_with_attrs(second_pos, [2]))

    assert jnp.allclose(batched.array[:2], first.array)
    assert jnp.allclose(batched.array[2:], second.array)


def test_all_pairs_long_range_encoding_padding_mask():
    valid_pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, 0.0]])
    padded_pos = np.concatenate([valid_pos, [[0.0, 0.0, 0.0]]])
    layer = gcnn.AllPairsLongRangeEncoding(l_max=2)
    expected = _long_range_features(layer, _graph_with_attrs(valid_pos, [2]))
    padded = _long_range_features(
        layer, _graph_with_attrs(padded_pos, [3], mask=[True, True, False])
    )

    assert jnp.allclose(padded.array[:2], expected.array)
    assert jnp.allclose(padded.array[2], 0.0)
    assert not jnp.any(jnp.isnan(padded.array))


@pytest.mark.parametrize("jit", (False, True))
def test_all_pairs_long_range_encoding_jit(jit):
    graph = _graph_with_attrs([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [2])
    layer = gcnn.AllPairsLongRangeEncoding(l_max=1)
    apply = jax.jit(layer.apply) if jit else layer.apply

    output = apply({}, jax.tree_util.tree_map(lambda x: x, graph))
    assert output.nodes[keys.ATTRIBUTES].irreps == e3j.Irreps("1x0e + 0e + 1o")
