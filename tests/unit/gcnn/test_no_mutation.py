"""Regression tests: graph functions and modules must not modify the graph they are given.

Writing outputs into the ``nodes``/``edges``/``globals`` dicts of the input graph (rather than into
copies) modifies the caller's graph.  When that graph is captured by a function being traced by
JAX (``jit``, ``grad``, ...), tracers end up stored in it and leak out of the transformation.
"""

from collections.abc import Callable

import e3nn_jax as e3j
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import pytest

import tensorial
from tensorial import gcnn
from tensorial.gcnn import keys
from tensorial.gcnn.atomic import keys as atomic_keys

R_MAX = 1.5
CART = tensorial.CartesianTensor("ij=ji", i="1e")


@pytest.fixture
def graph() -> jraph.GraphsTuple:
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    graph = gcnn.graph_from_points(
        pos,
        r_max=R_MAX,
        nodes={
            atomic_keys.ATOMIC_NUMBERS: np.array([1, 8, 1, 8]),
            keys.FEATURES: e3j.IrrepsArray("2x0e", jnp.ones((4, 2))),
            "encoded": e3j.IrrepsArray(CART.irreps, jnp.ones((4, CART.irreps.dim))),
        },
    )
    n_edge = graph.senders.shape[0]
    assert n_edge > 0
    return graph._replace(
        edges={
            **graph.edges,
            "vec": jnp.ones((n_edge, 3)),
            keys.FEATURES: e3j.IrrepsArray("2x0e", jnp.ones((n_edge, 2))),
            "encoded": e3j.IrrepsArray(CART.irreps, jnp.ones((n_edge, CART.irreps.dim))),
        }
    )


def _flax(module) -> Callable[[jraph.GraphsTuple], jraph.GraphsTuple]:
    def apply(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        params = module.init(jax.random.key(0), graph)
        return module.apply(params, graph)

    return apply


GRAPH_FUNCTIONS = {
    "with_edge_vectors": gcnn.with_edge_vectors,
    "NodewiseLinear": _flax(gcnn.NodewiseLinear(irreps_out="3x0e", out_field="out")),
    "NodewiseEmbedding": _flax(gcnn.NodewiseEmbedding({keys.POSITIONS: "1o"})),
    "NodewiseDecoding": _flax(gcnn.NodewiseDecoding({"decoded": CART}, in_field="encoded")),
    "EdgewiseLinear": _flax(gcnn.EdgewiseLinear(irreps_out="3x0e", out_field="out")),
    "EdgewiseEmbedding": _flax(gcnn.EdgewiseEmbedding({"vec": "1o"}, out_field="out")),
    "EdgewiseDecoding": _flax(gcnn.EdgewiseDecoding({"decoded": CART}, in_field="encoded")),
    "RadialBasisEdgeEmbedding": _flax(gcnn.RadialBasisEdgeEmbedding(r_max=R_MAX)),
    "SpeciesTransform": gcnn.atomic.SpeciesTransform([1, 8]),
}


def _snapshot(graph: jraph.GraphsTuple) -> dict[str, dict]:
    """Take a copy of the contents of the dicts in a graph, so we can check if they changed."""
    return {name: dict(getattr(graph, name)) for name in ("nodes", "edges", "globals")}


def _assert_unchanged(graph: jraph.GraphsTuple, snapshot: dict[str, dict]):
    for name, before in snapshot.items():
        after = getattr(graph, name)
        assert after.keys() == before.keys(), f"graph.{name} was modified"
        for key, value in before.items():
            assert after[key] is value, f"graph.{name}[{key!r}] was replaced"


@pytest.mark.parametrize("fn", GRAPH_FUNCTIONS.values(), ids=GRAPH_FUNCTIONS.keys())
def test_does_not_modify_input_graph(fn, graph: jraph.GraphsTuple):
    snapshot = _snapshot(graph)
    fn(graph)
    _assert_unchanged(graph, snapshot)


@pytest.mark.parametrize("fn", GRAPH_FUNCTIONS.values(), ids=GRAPH_FUNCTIONS.keys())
def test_no_tracer_leak_from_captured_graph(fn, graph: jraph.GraphsTuple):
    """A graph captured by a traced function (rather than passed to it) must not get tracers"""
    snapshot = _snapshot(graph)

    def energy(pos):
        # Only the nodes are copied, so the edges and globals dicts are shared with the captured
        # graph
        traced = graph._replace(nodes={**graph.nodes, keys.POSITIONS: pos})
        out = fn(traced)
        return sum(
            jnp.sum(e3j.as_irreps_array(leaf).array)
            for leaf in jax.tree.leaves(out, is_leaf=lambda x: isinstance(x, e3j.IrrepsArray))
            if jnp.issubdtype(jnp.asarray(getattr(leaf, "array", leaf)).dtype, jnp.floating)
        )

    jax.grad(energy)(jnp.asarray(graph.nodes[keys.POSITIONS]))

    _assert_unchanged(graph, snapshot)
    tracers = [leaf for leaf in jax.tree.leaves(graph) if isinstance(leaf, jax.core.Tracer)]
    assert not tracers


def test_no_tracer_leak_from_params(graph: jraph.GraphsTuple):
    """Taking the gradient w.r.t. a module's parameters, with the graph captured, must not leak"""
    module = gcnn.NodewiseLinear(irreps_out="3x0e", out_field="out")
    params = module.init(jax.random.key(0), graph)
    snapshot = _snapshot(graph)

    def loss(params_):
        return jnp.sum(module.apply(params_, graph).nodes["out"].array)

    jax.grad(loss)(params)

    _assert_unchanged(graph, snapshot)
    assert not [leaf for leaf in jax.tree.leaves(graph) if isinstance(leaf, jax.core.Tracer)]


def test_graph_from_points_does_not_modify_arguments():
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    nodes = {"species": np.array([1, 8])}
    graph_globals = {"energy": 1.0}

    gcnn.graph_from_points(
        pos, r_max=2, nodes=nodes, graph_globals=graph_globals, pbc=False, cell=None
    )

    assert nodes.keys() == {"species"}
    assert nodes["species"].shape == (2,)
    assert graph_globals == {"energy": 1.0}


def test_random_spatial_graph_does_not_modify_nodes_spec():
    def make_species(rng_key, num_nodes):
        return jax.random.randint(rng_key, (num_nodes,), 0, 3)

    nodes = {"species": make_species}
    graphs = gcnn.random.spatial_graph(jax.random.key(0), num_nodes=4, num_graphs=2, nodes=nodes)

    assert nodes == {"species": make_species}
    assert all(g.nodes["species"].shape == (4, 1) for g in graphs)
