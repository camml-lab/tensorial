import jax.numpy as jnp
import jraph
import numpy as np

from tensorial import gcnn


def test_transform_fn(cube_graph: jraph.GraphsTuple):
    new_pos = np.random.rand(cube_graph.n_node.item(), 3)
    ref_lengths = np.linalg.norm(
        new_pos[cube_graph.receivers] - new_pos[cube_graph.senders], axis=1
    ).reshape(-1, 1)

    # In only
    fn = gcnn.transform_fn(gcnn.with_edge_vectors, "nodes.positions")
    out_graph = fn(cube_graph, new_pos)
    assert np.allclose(out_graph.edges[gcnn.keys.EDGE_LENGTHS].array, ref_lengths)

    # In only with return_graph=True
    fn = gcnn.transform_fn(gcnn.with_edge_vectors, "nodes.positions", return_graphs=True)
    out_value, out_graph = fn(cube_graph, new_pos)
    assert isinstance(out_value, jraph.GraphsTuple)
    assert isinstance(out_graph, jraph.GraphsTuple)
    assert out_value is not out_graph
    assert jnp.allclose(out_graph.nodes["positions"], new_pos)

    # In and out
    fn = gcnn.transform_fn(
        gcnn.with_edge_vectors, "nodes.positions", outs=[("edges", gcnn.keys.EDGE_LENGTHS)]
    )

    edge_lengths = fn(cube_graph, new_pos)
    assert np.allclose(edge_lengths.array, ref_lengths)

    # In and out return graph
    fn = gcnn.transform_fn(
        gcnn.with_edge_vectors,
        "nodes.positions",
        outs=[("edges", gcnn.keys.EDGE_LENGTHS)],
        return_graphs=True,
    )

    edge_lengths, out_graph = fn(cube_graph, new_pos)
    assert isinstance(out_graph, jraph.GraphsTuple)
    assert np.allclose(edge_lengths.array, ref_lengths)

    # Out only
    fn = gcnn.transform_fn(gcnn.with_edge_vectors, outs=[("edges", gcnn.keys.EDGE_LENGTHS)])
    edge_lengths = fn(cube_graph)
    assert np.allclose(edge_lengths.array, cube_graph.edges[gcnn.keys.EDGE_LENGTHS].array)


def test_transform_fn_basic_replacement():
    # Define a dummy function that adds 1 to nodes
    def fn(graph, *args):
        new_nodes = graph.nodes + 1
        return graph._replace(nodes=new_nodes)

    graph = jraph.GraphsTuple(
        nodes=1, edges=0, globals=None, senders=None, receivers=None, n_node=None, n_edge=None
    )

    result = gcnn.transform_fn(fn, "nodes")(graph, 5)  # 5 replaces nodes
    assert result.nodes == 6  # 5 + 1


def test_transform_fn_with_outs():
    def fn(graph):
        return graph._replace(nodes=graph.nodes + 2)

    graph = jraph.GraphsTuple(
        nodes=3, edges=0, globals=None, senders=None, receivers=None, n_node=None, n_edge=None
    )

    result = gcnn.transform_fn(fn, outs=["nodes"])(graph)
    assert result == 5  # 3 + 2


def test_transform_fn_with_return_graphs():
    def fn(graph):
        return graph._replace(nodes=graph.nodes + 10)

    graph = jraph.GraphsTuple(
        nodes=2, edges=0, globals=None, senders=None, receivers=None, n_node=None, n_edge=None
    )

    vals, g_out = gcnn.transform_fn(fn, outs=["nodes"], return_graphs=True)(graph)
    assert g_out.nodes == 12
    assert vals == 12


def test_transform_fn_extra_args():
    def fn(graph, scale):
        return graph._replace(nodes=graph.nodes * scale)

    graph = jraph.GraphsTuple(
        nodes=4, edges=0, globals=None, senders=None, receivers=None, n_node=None, n_edge=None
    )

    result = gcnn.transform_fn(fn, outs=["nodes"])(graph, 3)
    assert result == 12


def test_adapt_with_kwargs(cube_graph):
    def fn(graph):
        pos = graph.nodes["positions"]
        return graph.globals["scale"] * (jnp.linalg.norm(pos) ** 2).sum()

    new_fn = gcnn.transform_fn(fn, "nodes.positions", scale="globals.scale")
    res = new_fn(cube_graph, cube_graph.nodes["positions"], scale=4.0)
    assert jnp.isclose(res, 96.0)
