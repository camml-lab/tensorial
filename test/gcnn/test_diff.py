import jax
import jax.numpy as jnp
import jraph
import pytest

import tensorial
from tensorial import gcnn
from tensorial.gcnn import _diff, experimental, keys


def test_invalid_output_indices_raises():
    with pytest.raises(ValueError, match="not in of"):
        _diff.SingleDerivative.create(
            of="globals.energy", wrt="nodes.positions:Iα", out="globals.energy:nonexistent_index"
        )


def test_output_index_inference():
    deriv = _diff.SingleDerivative.create(of="globals.energy", wrt="nodes.positions:Iα")
    assert deriv.out.indices == "Iα"


def energy_fn(graph_) -> jraph.GraphsTuple:
    graph_ = gcnn.with_edge_vectors(graph_, as_irreps_array=False)
    edge_vecs = tensorial.as_array(graph_.edges[keys.EDGE_VECTORS])
    return (
        experimental.update_graph(graph_)
        .set("globals.energy", sum(jax.vmap(jnp.dot, (0, 0))(edge_vecs, edge_vecs)))
        .get()
    )


@pytest.mark.parametrize("jit", [False, True])
def test_single_derivative_basic(jit):
    # Create a simple graph with two points
    graph = gcnn.graph_from_points(jnp.array([[0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]]), r_max=2.0)

    # Define the derivative: d(energy) / d(positions)
    diff = gcnn.diff(energy_fn, "globals.energy", wrt="nodes.positions:Iα", out=":Iα")
    if jit:
        diff = jax.jit(diff)

    # Evaluate the derivative with respect to new node positions
    new_positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    result = diff(graph, **{"nodes.positions": new_positions})

    # Check the shape and value (e.g., gradient magnitude)
    assert result.shape == (2, 3)  # Two nodes, 3 coordinates
    assert jnp.allclose(jnp.abs(result), 4.0, atol=1e-5), f"Unexpected derivative result: {result}"

    scale = -1.0
    diff = gcnn.diff(energy_fn, "globals.energy", wrt="nodes.positions:Iα", out=":Iα", scale=scale)
    if jit:
        diff = jax.jit(diff)
    result_2 = diff(graph, **{"nodes.positions": new_positions})
    assert jnp.allclose(
        result_2, scale * result, atol=1e-5
    ), f"Unexpected derivative result: {result}"

    diff = gcnn.diff(
        energy_fn, "globals.energy", wrt="nodes.positions:Iα", out=":Iα", return_graph=True
    )
    result_3, _ = diff(graph, **{"nodes.positions": new_positions})
    assert jnp.allclose(result, result_3)


@pytest.mark.parametrize("jit", [False, True])
def test_single_derivative_at(jit):
    # Create a simple graph with two points
    graph = gcnn.graph_from_points(jnp.array([[0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]]), r_max=2.0)

    # Evaluate the derivative with respect to new node positions
    new_positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

    # Define the derivative: d(energy) / d(positions)
    diff = gcnn.diff(
        energy_fn,
        "globals.energy",
        wrt="nodes.positions:Iα",
        at={"nodes.positions": new_positions},
        out=":Iα",
    )
    if jit:
        diff = jax.jit(diff)

    result = diff(graph)

    # Check the shape and value (e.g., gradient magnitude)
    assert result.shape == (2, 3)  # Two nodes, 3 coordinates
    assert jnp.allclose(jnp.abs(result), 4.0, atol=1e-5), f"Unexpected derivative result: {result}"


@pytest.mark.parametrize("jit", [True, False])
def test_derivative_of_fn(jit):
    # Create a simple graph with two points
    graph = gcnn.graph_from_points(jnp.array([[0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]]), r_max=2.0)

    energy_fn_ = gcnn.transform_fn(energy_fn, outs=["globals.energy"])
    # Define the derivative: d(energy) / d(positions)
    diff = gcnn.diff(energy_fn_, "", wrt="nodes.positions:Iα", out=":Iα")

    if jit:
        diff = jax.jit(diff)

    # Evaluate the derivative with respect to new node positions
    new_positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    result = diff(graph, **{"nodes.positions": new_positions})

    # Check the shape and value (e.g., gradient magnitude)
    assert result.shape == (2, 3)  # Two nodes, 3 coordinates
    assert jnp.allclose(jnp.abs(result), 4.0, atol=1e-5), f"Unexpected derivative result: {result}"


@pytest.mark.parametrize("jit", [False, True])
def test_multi_derivative(jit):
    def scale_energy_fn(graph_) -> jraph.GraphsTuple:
        graph_ = energy_fn(graph_)
        return (
            experimental.update_graph(graph_)
            .set("globals.energy", graph_.globals["scale"] * graph_.globals["energy"])
            .get()
        )

    graph = gcnn.graph_from_points(
        jnp.array([[0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]]), r_max=2.0, graph_globals={"scale": 2.0}
    )

    diff = gcnn.diff(
        scale_energy_fn,
        "globals.energy:",
        wrt=["nodes.positions:Ia", "globals.scale"],
    )

    if jit:
        diff = jax.jit(diff)

    pos = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    scale = 2.0
    res = diff(graph, **{"nodes.positions": pos, "globals.scale": scale})
    assert jnp.allclose(jnp.abs(res), 2.0 * scale)


@pytest.mark.parametrize("jit", [False, True])
def test_diff_numeric(jit):
    """Test taking the derivative of wrt to the same variable multiple times"""

    def scaled_energy(graph, scale: float):
        return scale * energy_fn(graph).globals["energy"]

    graph = gcnn.graph_from_points(
        jnp.array([[0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]]), r_max=2.0, graph_globals={"scale": 2.0}
    )

    diff = gcnn.diff(
        scaled_energy,
        # WRT argument 1 of the energy scale
        wrt=["nodes.positions:Ia", "1:"],
    )
    if jit:
        diff = jax.jit(diff)

    pos = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    scale = 2.0
    res = diff(graph, 2.0 * scale, **{"nodes.positions": pos})
    assert jnp.allclose(jnp.abs(res), 2.0 * scale)


@pytest.mark.parametrize("jit", [True, False])
def test_multi_same_deriv(jit):
    """Test taking the derivative of wrt to the same variable multiple times"""
    graph = gcnn.graph_from_points(jnp.array([[0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]]), r_max=2.0)

    diff = gcnn.diff(energy_fn, "globals.energy", wrt=["nodes.positions:Iα", "nodes.positions:Jα"])
    if jit:
        diff = jax.jit(diff)

    pos = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    res = diff(graph, **{"nodes.positions": pos})
    assert res.shape == (2, 2, 3)
    assert jnp.allclose(jnp.abs(res), 4.0)

    # Check rearranging output indices
    diff = gcnn.diff(
        energy_fn,
        "globals.energy:",
        wrt=["nodes.positions:Iα", "nodes.positions:Jα", "nodes.positions:Kα"],
        out=":IαJK",
    )
    if jit:
        diff = jax.jit(diff)

    pos = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    res = diff(graph, **{"nodes.positions": pos})
    assert res.shape == (2, 3, 2, 2)
    assert jnp.allclose(jnp.abs(res), 0.0)


@pytest.mark.parametrize("jit", [True, False])
def test_diff_reduce(jit):
    """Test taking the derivative of wrt to the same variable multiple times"""
    graph = gcnn.graph_from_points(jnp.array([[0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]]), r_max=2.0)

    diff = gcnn.diff(
        energy_fn,
        "globals.energy",
        wrt=["nodes.positions:Iα"],
        # For the output, we explicitly specify no indices i.e. a scalar so everything
        # should be reduced
        out=":",
    )
    if jit:
        diff = jax.jit(diff)

    pos = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    res = diff(graph, **{"nodes.positions": pos})
    assert res.shape == tuple()

    # The forces should be zero as there are only two particles and f_01 = -f_10
    assert jnp.allclose(jnp.abs(res), 0.0)


def test_graph_spec():
    spec = _diff.GraphEntrySpec.create("")
    assert spec.key_path == tuple()
    assert spec.indices is None

    spec = _diff.GraphEntrySpec.create(":")
    assert spec.key_path == tuple()
    assert spec.indices == ""

    spec = _diff.GraphEntrySpec.create(":ij")
    assert spec.key_path == tuple()
    assert spec.indices == "ij"

    spec = _diff.GraphEntrySpec.create("nodes.positions")
    assert spec.key_path == ("nodes", "positions")
    assert spec.indices is None

    spec = _diff.GraphEntrySpec.create("nodes.positions:")
    assert spec.key_path == ("nodes", "positions")
    assert spec.indices == ""

    spec = _diff.GraphEntrySpec.create("nodes.positions:ij")
    assert spec.key_path == ("nodes", "positions")
    assert spec.indices == "ij"
