import jax
import jax.numpy as jnp
import jraph
import pytest

from tensorial import gcnn


def test_rescale(rng_key):
    # Define a three node graph, each node has an integer as its feature.
    vals = jnp.array([[0.0], [1.0], [2.0]])
    shift = 12345.678
    scale = 5.76

    node_features = {"vals": vals}

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

    rescale = gcnn.Rescale(shift_fields="nodes.vals", shift=shift)
    params = rescale.init(rng_key, graph)
    out = rescale.apply(params, graph)
    assert jnp.all(out.nodes["vals"] == vals + shift)

    rescale = gcnn.Rescale(scale_fields="nodes.vals", scale=scale)
    params = rescale.init(rng_key, graph)
    out = rescale.apply(params, graph)
    assert jnp.all(out.nodes["vals"] == vals * scale)

    rescale = gcnn.Rescale(
        scale_fields="nodes.vals", shift_fields="nodes.vals", scale=scale, shift=shift
    )
    params = rescale.init(rng_key, graph)
    out = rescale.apply(params, graph)
    assert jnp.all(out.nodes["vals"] == vals * scale + shift)


@pytest.fixture
def linear_simple_graph():
    # A simple graph with 3 nodes, each assigned a "type" index in [0, 2)
    nodes = {
        "features": jnp.array([[1.0], [1.0], [1.0]]),  # (3, 1)
        "type": jnp.array([0, 1, 0]),  # (3,)
    }
    return jraph.GraphsTuple(
        nodes=nodes,
        edges={},
        senders=jnp.array([], dtype=jnp.int32),
        receivers=jnp.array([], dtype=jnp.int32),
        n_node=jnp.array([3]),
        n_edge=jnp.array([0]),
        globals={},
    )


def test_indexed_linear_output_shape(linear_simple_graph):
    module = gcnn.IndexedLinear(
        irreps_out="2x0e",
        num_types=2,
        index_field="nodes.type",
        field="nodes.features",
        out_field="nodes.out",
    )

    @jax.jit
    def forward(graph):
        return module.init_with_output(jax.random.PRNGKey(0), graph)[0]

    out_graph = forward(linear_simple_graph)

    assert "out" in out_graph.nodes
    assert out_graph.nodes["out"].shape == (3, 2)  # 2x0e gives dim 2
    assert out_graph.nodes["features"].shape == (3, 1)  # Original untouched


def test_indexed_linear_different_types_produce_different_outputs(linear_simple_graph):
    module = gcnn.IndexedLinear(
        irreps_out="1x0e",
        num_types=2,
        index_field="nodes.type",
        field="nodes.features",
        out_field="nodes.out",
    )

    params = module.init(jax.random.PRNGKey(42), linear_simple_graph)
    out_graph = module.apply(params, linear_simple_graph)

    out = out_graph.nodes["out"]
    assert out.shape == (3, 1)

    # Since types are [0, 1, 0], the 0th and 2nd node should be equal, 1st should be different
    assert jnp.allclose(out[0].array, out[2].array, atol=1e-5)
    assert not jnp.allclose(out[0].array, out[1].array, atol=1e-5)


def test_indexed_linear_inplace_overwrite(linear_simple_graph):
    module = gcnn.IndexedLinear(
        irreps_out="1x0e",
        num_types=2,
        index_field="nodes.type",
        field="nodes.features",  # No out_field → overwrite
    )

    params = module.init(jax.random.PRNGKey(0), linear_simple_graph)
    out_graph = module.apply(params, linear_simple_graph)

    # Should have replaced "features" field
    assert "features" in out_graph.nodes
    assert out_graph.nodes["features"].shape == (3, 1)


@pytest.fixture
def rescale_simple_graph():
    """Creates a minimal graph with node features and types for testing."""
    return jraph.GraphsTuple(
        nodes={"features": jnp.array([[1.0], [2.0], [3.0]]), "type": jnp.array([0, 1, 0])},
        edges={},
        senders=jnp.array([], dtype=jnp.int32),
        receivers=jnp.array([], dtype=jnp.int32),
        globals={},
        n_node=jnp.array([3]),
        n_edge=jnp.array([0]),
    )


def test_indexed_rescale_constant_scaling(rescale_simple_graph):
    """Test rescaling with constant user-provided scales and shifts."""
    module = gcnn.IndexedRescale(
        num_types=2,
        index_field="nodes.type",
        field="nodes.features",
        shifts=jnp.array([10.0, -1.0]),
        scales=jnp.array([[2.0], [3.0]]),
    )
    params = module.init(jax.random.PRNGKey(0), rescale_simple_graph)
    out_graph = module.apply(params, rescale_simple_graph)

    expected = jnp.array([[1.0 * 2 + 10], [2.0 * 3 - 1], [3.0 * 2 + 10]])
    assert jnp.allclose(out_graph.nodes["features"], expected, atol=1e-5)


def test_indexed_rescale_learned_scaling(rescale_simple_graph):
    """Test rescaling with learned parameters."""
    module = gcnn.IndexedRescale(
        num_types=2,
        index_field="nodes.type",
        field="nodes.features",
    )
    params = module.init(jax.random.PRNGKey(0), rescale_simple_graph)
    out_graph = module.apply(params, rescale_simple_graph)

    # Output shape should match input
    assert out_graph.nodes["features"].shape == (3, 1)


def test_indexed_rescale_out_field(rescale_simple_graph):
    """Test that output can be written to a different field."""
    module = gcnn.IndexedRescale(
        num_types=2,
        index_field="nodes.type",
        field="nodes.features",
        out_field="nodes.transformed",
        shifts=jnp.array([0.0, 0.0]),
        scales=jnp.array([[1.0], [1.0]]),
    )
    params = module.init(jax.random.PRNGKey(0), rescale_simple_graph)
    out_graph = module.apply(params, rescale_simple_graph)

    # Original field should remain unchanged
    assert jnp.allclose(out_graph.nodes["features"], rescale_simple_graph.nodes["features"])
    # Transformed field should be identical to the input (identity rescale)
    assert jnp.allclose(out_graph.nodes["transformed"], rescale_simple_graph.nodes["features"])
