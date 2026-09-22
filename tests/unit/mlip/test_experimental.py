import jax.numpy as jnp
import jraph
import pytest

from tensorial.mlip.experimental import minimize_fn


@pytest.fixture
def graph() -> jraph.GraphsTuple:
    """Simple graph with no edges and a vector of parameters stored in globals."""
    n_nodes = 5
    return jraph.GraphsTuple(
        nodes={"pos": jnp.zeros((n_nodes, 3))},
        n_node=n_nodes,
        n_edge=0,
        edges={},
        senders=jnp.array([], dtype=jnp.int32),
        receivers=jnp.array([], dtype=jnp.int32),
        globals={"params": jnp.ones((2, 2))},
    )


@pytest.fixture
def objective():
    """Quadratic objective, E(x) = sum(0.5 * A * x**2 + b * x), stored as globals.energy."""
    a = jnp.array([[0.5, 1.0], [1.5, 2.0]])
    b = jnp.array([[1.0, -1.0], [1.0, -1.0]])

    def fun(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        params = graph.globals["params"]
        energy = jnp.sum(0.5 * a * params**2 + b * params)
        return graph._replace(globals={**graph.globals, "energy": energy})

    return fun


def test_minimize_fn(graph: jraph.GraphsTuple, objective):
    x0 = graph.globals["params"]
    minim = minimize_fn(objective, what="globals.energy", wrt="globals.params")
    res = minim(graph, x0, method="bfgs", options={"maxiter": 100})
    assert res.success
    assert res.x.shape == x0.shape
    # The exact minimum is x* = -b / A, element wise
    expected = -jnp.array([[1.0, -1.0], [1.0, -1.0]]) / jnp.array([[0.5, 1.0], [1.5, 2.0]])
    assert jnp.allclose(res.x, expected, atol=1e-4)


def test_minimize_fn_jac_shape(graph: jraph.GraphsTuple, objective):
    x0 = graph.globals["params"]
    minim = minimize_fn(objective, what="globals.energy", wrt="globals.params")
    res = minim(graph, x0, method="bfgs", options={"maxiter": 100})
    assert res.jac.shape == x0.shape
    # Converged, so the gradient should be (approximately) zero
    assert jnp.allclose(res.jac, 0.0, atol=1e-4)
