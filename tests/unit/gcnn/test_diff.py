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
    new_pos = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    result = diff(graph, **{"nodes.positions": new_pos})

    # Check the shape and value (e.g., gradient magnitude)
    assert result.shape == (2, 3)  # Two nodes, 3 coordinates
    assert jnp.allclose(jnp.abs(result), 4.0, atol=1e-5), f"Unexpected derivative result: {result}"

    scale = -1.0
    diff = gcnn.diff(energy_fn, "globals.energy", wrt="nodes.positions:Iα", out=":Iα", scale=scale)
    if jit:
        diff = jax.jit(diff)
    result_2 = diff(graph, **{"nodes.positions": new_pos})
    assert jnp.allclose(
        result_2, scale * result, atol=1e-5
    ), f"Unexpected derivative result: {result}"

    diff = gcnn.diff(
        energy_fn, "globals.energy", wrt="nodes.positions:Iα", out=":Iα", return_graph=True
    )
    result_3, _ = diff(graph, **{"nodes.positions": new_pos})
    assert jnp.allclose(result, result_3)


@pytest.mark.parametrize("jit", [False, True])
def test_single_derivative_at(jit):
    # Create a simple graph with two points
    graph = gcnn.graph_from_points(jnp.array([[0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]]), r_max=2.0)

    # Evaluate the derivative with respect to new node positions
    new_pos = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

    # Define the derivative: d(energy) / d(positions)
    diff = gcnn.diff(
        energy_fn,
        "globals.energy",
        wrt="nodes.positions:Iα",
        at={"nodes.positions": new_pos},
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


def moment_energy_fn(graph_) -> jraph.GraphsTuple:
    """E = sum_k (mu_k . r_k)(B . r_k), chosen because the mixed second derivative

        d2E / dmu_{k,i} dB_j = r_{k,i} r_{k,j}

    is known in closed form, and because the two levels want opposite modes: the inner
    derivative has a scalar output and 3N inputs, the outer has 3 inputs and 3N outputs.
    """
    pos = tensorial.as_array(graph_.nodes[keys.POSITIONS])
    mu = tensorial.as_array(graph_.nodes["mu"])
    field = tensorial.as_array(graph_.globals["field"])  # (n_graph, 3)
    # Shaped (n_graph,) so that the 'g' index in the specs below is real
    energy = jnp.sum((mu * pos).sum(-1) * (pos @ field[0])).reshape(1)
    return experimental.update_graph(graph_).set("globals.energy", energy).get()


def moment_graph():
    pos = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.5, -0.5],
            [-1.0, 0.25, 1.0],
            [0.5, -1.0, 0.25],
            [-0.25, 0.75, -1.0],
        ]
    )
    return gcnn.graph_from_points(
        pos,
        r_max=4.0,
        nodes={"mu": jnp.zeros_like(pos)},
        graph_globals={"field": jnp.zeros((1, 3))},
    )


def test_auto_picks_reverse_for_scalar_output():
    """Many inputs, one output: reverse mode gets the whole gradient in a single pass."""
    graph = moment_graph()
    deriv = _diff.SingleDerivative.create(
        of="globals.energy:g", wrt="nodes.positions:Iα", out=":Iα"
    )
    # 'g' is summed away by _pre_process, so a scalar is what actually gets differentiated
    assert deriv.differentiated_size(graph) == 1
    assert deriv.choose_mode(graph, graph.nodes[keys.POSITIONS]) == "rev"


def test_auto_picks_forward_for_few_inputs():
    """Few inputs, many outputs: forward mode costs one pass per input component."""
    graph = moment_graph()
    deriv = _diff.SingleDerivative.create(of="nodes.mu:Iγ", wrt="globals.field:gα", out=":Iγα")
    assert deriv.differentiated_size(graph) == 3 * graph.n_node.sum()  # 3 per node
    assert deriv.choose_mode(graph, graph.globals["field"]) == "fwd"


def test_auto_sizes_a_chain_intermediate_from_its_labels():
    """A chain's intermediate has no key path, so its extents come from the other links.

    Regression: sizing it from the spec alone made the node index look like a Cartesian
    one, which flipped the outer derivative to reverse mode once n_graphs grew.
    """
    graph = moment_graph()
    n_nodes = int(graph.n_node.sum())
    deriv = _diff.MultiDerivative.create(
        of="globals.energy:g", wrt=["nodes.mu:Iα", "globals.field:gβ"], out=":Iαβ"
    )
    args = (graph.nodes["mu"], graph.globals["field"])
    extents = deriv.index_extents(graph, args)
    assert extents["I"] == n_nodes
    assert extents["α"] == 3

    inner, outer = deriv[0], deriv[1]
    assert inner.differentiated_size(graph, extents) == 1  # scalar energy
    assert outer.differentiated_size(graph, extents) == 3 * n_nodes
    # Without them, 'I' is indistinguishable from a Cartesian index
    assert outer.differentiated_size(graph) == _diff.CARTESIAN_DIM**2
    assert inner.choose_mode(graph, args[0], extents) == "rev"
    assert outer.choose_mode(graph, args[1], extents) == "fwd"


def test_auto_sizes_survive_batching():
    """The mis-sizing only changes the chosen mode once the batch is big enough.

    Estimating an intermediate's node index as Cartesian caps the outer derivative's output
    at CARTESIAN_DIM ** 2 = 9.  Its input is 3 * n_graphs, so from four graphs on the estimate
    says reverse is cheaper when it is not -- worth 15x peak memory on a real model.
    """
    graph = jraph.batch([moment_graph() for _ in range(4)])
    deriv = _diff.MultiDerivative.create(
        of="globals.energy:g", wrt=["nodes.mu:Iα", "globals.field:gβ"], out=":Iαβ"
    )
    args = (graph.nodes["mu"], graph.globals["field"])
    extents = deriv.index_extents(graph, args)
    outer = deriv[1]

    assert outer.differentiated_size(graph, extents) == 3 * int(graph.n_node.sum())
    assert outer.choose_mode(graph, args[1], extents) == "fwd"
    # What the un-propagated estimate produced, kept to pin the regression
    assert outer.differentiated_size(graph) == _diff.CARTESIAN_DIM**2
    assert outer.choose_mode(graph, args[1]) == "rev"


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("mode", ["rev", "fwd", "auto"])
def test_auto_matches_explicit_modes(jit, mode):
    """Every mode must agree with the analytic mixed derivative r_k (x) r_k."""
    graph = moment_graph()
    pos = tensorial.as_array(graph.nodes[keys.POSITIONS])

    diff = gcnn.diff(
        moment_energy_fn,
        "globals.energy:g",
        wrt=["nodes.mu:Iα", "globals.field:gβ"],
        out=":Iαβ",
        mode=mode,
    )
    if jit:
        diff = jax.jit(diff)

    res = diff(graph, jnp.zeros_like(pos), jnp.zeros((1, 3)))
    expected = jax.vmap(jnp.outer)(pos, pos)
    assert res.shape == expected.shape
    assert jnp.allclose(res, expected, atol=1e-6), f"{mode} gave {res}, expected {expected}"


def test_auto_mixes_modes_along_a_chain(caplog):
    """The whole point of "auto": the two links of one chain choose differently."""
    import logging

    graph = moment_graph()
    pos = tensorial.as_array(graph.nodes[keys.POSITIONS])
    diff = gcnn.diff(
        moment_energy_fn,
        "globals.energy:g",
        wrt=["nodes.mu:Iα", "globals.field:gβ"],
        out=":Iαβ",
        mode="auto",
    )
    with caplog.at_level(logging.DEBUG, logger="tensorial.gcnn._diff"):
        diff(graph, jnp.zeros_like(pos), jnp.zeros((1, 3)))

    picked = [rec.getMessage() for rec in caplog.records if "choosing" in rec.getMessage()]
    assert any("choosing rev" in msg for msg in picked), picked
    assert any("choosing fwd" in msg for msg in picked), picked


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="mode must be one of"):
        gcnn.diff(energy_fn, "globals.energy", wrt="nodes.positions:Iα", mode="reverse")
