import jax
import jax.numpy as jnp
import jraph
import pytest

import tensorial
from tensorial import gcnn
from tensorial.gcnn import experimental, keys


def test_grads():
    def get_norms(pos, graph):
        # Have to do a strange thing here where we set the positions (even though they
        # already exist) to make this function a function of the positions that we can then
        # take derivatives of
        graph.nodes[keys.POSITIONS] = pos
        graph = gcnn.with_edge_vectors(graph)
        return tensorial.as_array(graph.edges[keys.EDGE_LENGTHS])[0, 0]

    pos = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

    graph = gcnn.graph_from_points(pos, r_max=2.0, np_=jnp)
    norms = get_norms(pos, graph)
    assert jnp.allclose(norms, jnp.sqrt(3.0))

    grads = jax.grad(get_norms)(pos, graph)
    assert jnp.allclose(jnp.abs(grads), 1.0 / jnp.sqrt(3.0))


def test_grad_module(rng_key):
    def get_energy(graph_):
        graph_ = gcnn.with_edge_vectors(graph_)
        return graph_

    pos = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    g1 = gcnn.graph_from_points(pos, r_max=2.0)
    g2 = gcnn.graph_from_points(pos, r_max=2.0)
    graph = jraph.batch([g1, g2])

    grad = gcnn.Grad(
        get_energy,
        of=f"edges.{keys.EDGE_LENGTHS}",
        wrt=f"nodes.{keys.POSITIONS}",
    )

    params = grad.init(rng_key, graph)
    res = grad.apply(params, graph)
    assert jnp.allclose(
        jnp.abs(res.nodes[f"d{keys.EDGE_LENGTHS}/d{keys.POSITIONS}"]), 2.0 / jnp.sqrt(3.0)
    )


def test_grad_vectors(rng_key):
    def get_energy(g):
        edge_vecs = tensorial.as_array(g.edges[keys.EDGE_VECTORS])
        gbals = g.globals
        gbals["energy"] = sum(jnp.linalg.norm(edge_vecs, axis=1) ** 2)
        return g._replace(globals=gbals)

    pos = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    graph = gcnn.graph_from_points(pos, r_max=2.0)
    graph = gcnn.with_edge_vectors(graph, as_irreps_array=False)

    # This time, let's test the grad() partial
    grad = gcnn.grad(of="globals.energy", wrt=f"edges.{keys.EDGE_VECTORS}")(get_energy)
    res = grad(graph, graph.edges[keys.EDGE_VECTORS])
    assert jnp.allclose(jnp.abs(res), 2.0)


def test_invalid_output_indices_raises():
    with pytest.raises(ValueError, match="not in of"):
        gcnn.experimental.derivatives.SingleDerivative.create(
            of="globals.energy", wrt="nodes.positions:Iα", out="globals.energy:nonexistent_index"
        )


def test_output_index_inference():
    deriv = gcnn.experimental.derivatives.SingleDerivative.create(
        of="globals.energy", wrt="nodes.positions:Iα"
    )
    assert deriv.out.indices == "Iα"


def energy_fn(graph_) -> jraph.GraphsTuple:
    graph_ = gcnn.with_edge_vectors(graph_, as_irreps_array=False)
    edge_vecs = tensorial.as_array(graph_.edges[keys.EDGE_VECTORS])
    return (
        experimental.update_graph(graph_)
        .set("globals.energy", sum(jax.vmap(jnp.dot, (0, 0))(edge_vecs, edge_vecs)))
        .get()
    )


@pytest.mark.parametrize("jit", [True, False])
def test_single_derivative_basic(jit):
    # Create a simple graph with two points
    graph = gcnn.graph_from_points(jnp.array([[0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]]), r_max=2.0)

    # Define the derivative: d(energy) / d(positions)
    diff = gcnn.experimental.diff(energy_fn, "globals.energy", wrt="nodes.positions:Iα", out=":Iα")
    if jit:
        diff = jax.jit(diff)

    # Evaluate the derivative with respect to new node positions
    new_positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    result = diff(graph, new_positions)

    # Check the shape and value (e.g., gradient magnitude)
    assert result.shape == (2, 3)  # Two nodes, 3 coordinates
    assert jnp.allclose(jnp.abs(result), 4.0, atol=1e-5), f"Unexpected derivative result: {result}"

    scale = -1.0
    diff = gcnn.experimental.diff(
        energy_fn, "globals.energy", wrt="nodes.positions:Iα", out=":Iα", scale=scale
    )
    if jit:
        diff = jax.jit(diff)
    result_2 = diff(graph, new_positions)
    assert jnp.allclose(
        result_2, scale * result, atol=1e-5
    ), f"Unexpected derivative result: {result}"

    diff = gcnn.experimental.diff(
        energy_fn, "globals.energy", wrt="nodes.positions:Iα", out=":Iα", return_graph=True
    )
    result_3, _ = diff(graph, new_positions)
    assert jnp.allclose(result, result_3)


@pytest.mark.parametrize("jit", [True, False])
def test_derivative_of_fn(jit):
    # Create a simple graph with two points
    graph = gcnn.graph_from_points(jnp.array([[0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]]), r_max=2.0)

    energy_fn_ = gcnn.transform_fn(energy_fn, outs=["globals.energy"])
    # Define the derivative: d(energy) / d(positions)
    diff = gcnn.experimental.diff(energy_fn_, "", wrt="nodes.positions:Iα", out=":Iα")

    if jit:
        diff = jax.jit(diff)

    # Evaluate the derivative with respect to new node positions
    new_positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    result = diff(graph, new_positions)

    # Check the shape and value (e.g., gradient magnitude)
    assert result.shape == (2, 3)  # Two nodes, 3 coordinates
    assert jnp.allclose(jnp.abs(result), 4.0, atol=1e-5), f"Unexpected derivative result: {result}"


@pytest.mark.parametrize("jit", [True, False])
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

    diff = gcnn.experimental.diff(
        scale_energy_fn,
        "globals.energy:",
        wrt=["nodes.positions:Ia", "globals.scale"],
        # out=":Iαβ",
    )

    if jit:
        diff = jax.jit(diff)

    pos = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    scale = 2.0
    res = diff(graph, pos, scale)
    assert jnp.allclose(jnp.abs(res), 2.0 * scale)


@pytest.mark.parametrize("jit", [True, False])
def test_multi_same_deriv(jit):
    """Test taking the derivative of wrt to the same variable multiple times"""
    graph = gcnn.graph_from_points(jnp.array([[0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]]), r_max=2.0)

    diff = gcnn.experimental.diff(
        energy_fn,
        "globals.energy",
        wrt=["nodes.positions:Iα", "nodes.positions:Jα"],
    )
    if jit:
        diff = jax.jit(diff)

    pos = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    res = diff(graph, pos)
    assert res.shape == (2, 2, 3)
    assert jnp.allclose(jnp.abs(res), 4.0)

    # Check rearranging output indices
    diff = gcnn.experimental.diff(
        energy_fn,
        "globals.energy:",
        wrt=["nodes.positions:Iα", "nodes.positions:Jα", "nodes.positions:Kα"],
        out=":IαJK",
    )
    if jit:
        diff = jax.jit(diff)

    pos = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    res = diff(graph, pos)
    assert res.shape == (2, 3, 2, 2)
    assert jnp.allclose(jnp.abs(res), 0.0)


@pytest.mark.parametrize("jit", [True, False])
def test_diff_reduce(jit):
    """Test taking the derivative of wrt to the same variable multiple times"""
    graph = gcnn.graph_from_points(jnp.array([[0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]]), r_max=2.0)

    diff = gcnn.experimental.diff(
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
    res = diff(graph, pos)
    assert res.shape == tuple()

    # The forces should be zero as there are only two particles and f_01 = -f_10
    assert jnp.allclose(jnp.abs(res), 0.0)


def test_graph_spec():
    spec = gcnn.experimental.derivatives.GraphEntrySpec.create("")
    assert spec.key_path == tuple()
    assert spec.indices is None

    spec = gcnn.experimental.derivatives.GraphEntrySpec.create(":")
    assert spec.key_path == tuple()
    assert spec.indices == ""

    spec = gcnn.experimental.derivatives.GraphEntrySpec.create(":ij")
    assert spec.key_path == tuple()
    assert spec.indices == "ij"

    spec = gcnn.experimental.derivatives.GraphEntrySpec.create("nodes.positions")
    assert spec.key_path == ("nodes", "positions")
    assert spec.indices is None

    spec = gcnn.experimental.derivatives.GraphEntrySpec.create("nodes.positions:")
    assert spec.key_path == ("nodes", "positions")
    assert spec.indices == ""

    spec = gcnn.experimental.derivatives.GraphEntrySpec.create("nodes.positions:ij")
    assert spec.key_path == ("nodes", "positions")
    assert spec.indices == "ij"
