import jax
import jax.numpy as jnp
import pytest

from tensorial import gcnn
from tensorial.gcnn import _common


def test_reduce_jitting(cube_graph):
    reduced = jax.jit(gcnn.reduce, static_argnums=(1, 2))(cube_graph, "nodes.positions")

    assert reduced.shape == (1, 3)
    assert jnp.allclose(reduced, 0.0)  # The cube is centred at the origin


@pytest.mark.parametrize("op", ["sum", "mean", "min", "max"])
def test_reduce_ops(cube_graph, op):
    reduced = gcnn.reduce(cube_graph, "nodes.positions", reduction=op)

    assert reduced.shape == (1, 3)

    res = getattr(jnp, op)(cube_graph.nodes["positions"], axis=0)
    assert jnp.allclose(reduced, res)


def test_reduce_value_types(cube_graph):
    # As dictionary
    reduced = gcnn.reduce(cube_graph._asdict(), "nodes.positions")
    assert reduced.shape == (1, 3)
    assert jnp.allclose(reduced, 0.0)  # The cube is centred at the origin


def test_red(cube_graph):
    pos = _common._reduce(cube_graph.nodes["positions"], cube_graph.n_node)

    assert pos.shape == (1, 3)
    assert jnp.allclose(pos, 0.0)

    with pytest.raises(jax.errors.ConcretizationTypeError):
        # If the number of segments is not passed as a static value, then this reduction will
        # not have a fixed shape, and therefore cannot be JITted
        jax.jit(_common._reduce)(cube_graph.nodes["positions"], cube_graph.n_node)

    # If the number of segments is not passed as a static value, then this reduction will
    # not have a fixed shape, and therefore cannot be JITted
    pos = jax.jit(_common._reduce, static_argnames="num_segments")(
        cube_graph.nodes["positions"],
        cube_graph.n_node,
        num_segments=cube_graph.nodes["positions"].shape[0],
    )

    assert pos.shape == (1, 3)
    assert jnp.allclose(pos, 0.0)
