import jraph
import numpy as np

from tensorial import gcnn


def test_transform_fn(cube_graph: jraph.GraphsTuple):
    new_pos = np.random.rand(cube_graph.n_node.item(), 3)
    ref_lengths = np.linalg.norm(
        new_pos[cube_graph.receivers] - new_pos[cube_graph.senders], axis=1
    ).reshape(-1, 1)

    # In only
    fn = gcnn.transform_fn(
        gcnn.with_edge_vectors,
        "nodes.positions",
    )
    out_graph = fn(cube_graph, new_pos)
    assert np.allclose(out_graph.edges[gcnn.keys.EDGE_LENGTHS].array, ref_lengths)

    # In only with graph (should do nothing)
    fn = gcnn.transform_fn(
        gcnn.with_edge_vectors,
        "nodes.positions",
        return_graphs=True,
    )
    out_graph = fn(cube_graph, new_pos)
    assert isinstance(out_graph, jraph.GraphsTuple)

    # In and out
    fn = gcnn.transform_fn(
        gcnn.with_edge_vectors,
        "nodes.positions",
        outs=[("edges", gcnn.keys.EDGE_LENGTHS)],
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
    assert np.allclose(edge_lengths.array, ref_lengths)
    assert isinstance(out_graph, jraph.GraphsTuple)

    # Out only
    fn = gcnn.transform_fn(
        gcnn.with_edge_vectors,
        outs=[("edges", gcnn.keys.EDGE_LENGTHS)],
    )
    edge_lengths = fn(cube_graph)
    assert np.allclose(edge_lengths.array, cube_graph.edges[gcnn.keys.EDGE_LENGTHS].array)
