import jax
import jax.numpy as jnp
import pytest

from tensorial.gcnn import graph_ops


def test_reduce_jitting(cube_graph):
    reduced = jax.jit(graph_ops.graph_segment_reduce, static_argnums=(1, 2))(
        cube_graph, "nodes.positions", reduction="sum"
    )

    assert reduced.shape == (1, 3)
    assert jnp.allclose(reduced, 0.0)  # The cube is centred at the origin


@pytest.mark.parametrize("op", ["sum", "mean", "min", "max"])
def test_reduce_ops(cube_graph, op):
    reduced = graph_ops.graph_segment_reduce(cube_graph, "nodes.positions", reduction=op)

    assert reduced.shape == (1, 3)

    res = getattr(jnp, op)(cube_graph.nodes["positions"], axis=0)
    assert jnp.allclose(reduced, res)


def test_reduce_value_types(cube_graph):
    # As dictionary
    reduced = graph_ops.graph_segment_reduce(cube_graph._asdict(), "nodes.positions")
    assert reduced.shape == (1, 3)
    assert jnp.allclose(reduced, 0.0)  # The cube is centred at the origin


def test_red(cube_graph):
    pos = graph_ops.segment_sum(cube_graph.nodes["positions"], cube_graph.n_node)
    assert pos.shape == (1, 3)
    assert jnp.allclose(pos, 0.0)

    # JAX jit
    pos = jax.jit(graph_ops.segment_sum)(cube_graph.nodes["positions"], cube_graph.n_node)
    assert pos.shape == (1, 3)
    assert jnp.allclose(pos, 0.0)
