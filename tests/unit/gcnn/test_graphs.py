import functools

import e3nn_jax as e3j
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import pytest

from tensorial import gcnn
from tensorial.gcnn import _spatial, keys


def test_graph_from_points():
    # Check that 1D
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    gcnn.graph_from_points(pos, r_max=2)

    # and 2D work
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    gcnn.graph_from_points(pos, r_max=2)


def test_graph_from_points_node_shape_mismatch():
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="foo"):
        gcnn.graph_from_points(pos, r_max=2, nodes={"foo": np.array([1.0, 2.0, 3.0])})


def test_graph_from_points_nodes():
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    nodes = {"species": np.array([1, 8]), "features": np.array([[1.0], [2.0]])}
    graph = gcnn.graph_from_points(pos, r_max=2, nodes=nodes)

    # 1D attributes are expanded to [N, 1] (unless they are a mask), higher-dim ones unchanged
    assert graph.nodes["species"].shape == (2, 1)
    np.testing.assert_array_equal(graph.nodes["species"], [[1], [8]])
    np.testing.assert_array_equal(graph.nodes["features"], [[1.0], [2.0]])


def test_graph_from_points_cell_without_pbc_raises():
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="without PBCs"):
        gcnn.graph_from_points(pos, r_max=2, cell=np.eye(3))


def test_graph_from_points_pbc_wrong_length():
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="length 3"):
        gcnn.graph_from_points(pos, r_max=2, pbc=(True, False))


def test_graph_from_points_bool_pbc():
    pos = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    graph = gcnn.graph_from_points(pos, r_max=0.6, cell=np.eye(3), pbc=True)
    np.testing.assert_array_equal(graph.globals[keys.PBC].astype(bool), [[True, True, True]])


def test_graph_from_points_fractional_positions_requires_cell():
    pos = np.array([[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]])
    with pytest.raises(ValueError, match="Unit cell"):
        gcnn.graph_from_points(pos, r_max=1.0, fractional_positions=True)


def test_graph_from_points_fractional_positions():
    cell = np.eye(3) * 2.0
    frac = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    expected = frac @ cell
    graph = gcnn.graph_from_points(
        list(frac), r_max=2.0, cell=cell, pbc=True, fractional_positions=True
    )
    np.testing.assert_allclose(graph.nodes[keys.POSITIONS], expected)


def test_graph_from_points_graph_globals():
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    graph = gcnn.graph_from_points(pos, r_max=2.0, graph_globals={"batch_idx": np.array([3])})

    # globals are expanded to [1, ...]
    np.testing.assert_array_equal(graph.globals["batch_idx"], [[3]])


def test_graph_from_points_open_boundary():
    r_max = 1.0
    pos = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])

    graph = gcnn.graph_from_points(pos, r_max=r_max)
    assert graph.n_edge.item() == 2

    # Self interaction shouldn't make a difference here but strict self interaction should
    for self_interaction in (True, False):
        graph = gcnn.graph_from_points(
            pos, r_max=r_max, self_interaction=self_interaction, strict_self_interaction=True
        )
        # 0 -> 0, 0 -> 1, 1 -> 1
        assert graph.n_edge.item() == 4


def test_graph_from_points_periodic():
    r_max = 0.6
    pos = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    cell = np.eye(3)

    graph = gcnn.graph_from_points(
        pos,
        r_max=r_max,
        cell=cell,
        pbc=False,
        self_interaction=False,
        strict_self_interaction=False,
    )
    assert graph.n_edge == 2

    graph = gcnn.graph_from_points(
        pos,
        r_max=r_max,
        cell=cell,
        pbc=(True, False, False),
        self_interaction=True,
        strict_self_interaction=False,
    )
    assert graph.n_edge.item() == 4

    graph = gcnn.graph_from_points(
        pos,
        r_max=r_max,
        cell=cell,
        pbc=(True, True, True),
        self_interaction=True,
        strict_self_interaction=False,
    )
    assert graph.n_edge.item() == 4

    graph = gcnn.graph_from_points(
        pos,
        r_max=r_max,
        cell=cell,
        pbc=(True, True, True),
        self_interaction=True,
        strict_self_interaction=True,
    )
    assert graph.n_edge.item() == 6


@pytest.mark.parametrize("with_lengths", (True, False))
def test_with_edge_vectors(with_lengths):
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    graph = gcnn.graph_from_points(pos, r_max=2)
    graph = gcnn.with_edge_vectors(graph, with_lengths=with_lengths)

    assert graph.n_node[0].item() == 2
    # Both way edges
    assert graph.n_edge[0].item() == 2
    assert len(graph.edges[gcnn.keys.EDGE_VECTORS]) == 2
    if with_lengths:
        assert len(graph.edges[gcnn.keys.EDGE_LENGTHS]) == 2
    else:
        assert gcnn.keys.EDGE_LENGTHS not in graph.edges


def test_with_edge_vectors_periodic():
    r_max = 0.9
    pos = np.array([[0.25, 0.0, 0.0], [0.75, 0.0, 0.0]])
    cell = np.eye(3) * 1.0

    graph = gcnn.graph_from_points(
        pos,
        r_max=r_max,
        cell=cell,
        pbc=True,
        self_interaction=True,
        strict_self_interaction=False,
    )
    graph = gcnn.with_edge_vectors(graph)

    # with PBCs every atom is a neighbour of its image, so we expect 4 edges
    assert graph.n_edge.item() == 4
    for edge in graph.edges[keys.EDGE_VECTORS].array:
        assert jnp.linalg.norm(edge) < r_max

    # the edge vectors include the cell shifts: one edge connects node 0 to the
    # periodic image of node 1 in the +x direction
    for shift in (0.0, 1.0):
        expected = pos[1] - cell[0] * shift - pos[0]
        assert jnp.linalg.norm(expected) < r_max
        assert jnp.any(
            jnp.all(jnp.abs(graph.edges[keys.EDGE_VECTORS].array - expected) < 1e-8, axis=-1)
        )


@pytest.mark.parametrize("as_irreps_array", (True, False))
def test_with_edge_vectors_not_irreps_array(as_irreps_array):
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    graph = gcnn.graph_from_points(pos, r_max=2)
    graph = gcnn.with_edge_vectors(graph, as_irreps_array=as_irreps_array)

    edge_vecs = graph.edges[keys.EDGE_VECTORS]
    if as_irreps_array:
        assert isinstance(edge_vecs, e3j.IrrepsArray)
    else:
        assert not isinstance(edge_vecs, e3j.IrrepsArray)
    np.testing.assert_allclose(
        np.asarray(edge_vecs.array if isinstance(edge_vecs, e3j.IrrepsArray) else edge_vecs),
        [[1, 0, 0], [-1, 0, 0]],
    )

    if as_irreps_array:
        assert isinstance(graph.edges[keys.EDGE_LENGTHS], e3j.IrrepsArray)
    lengths = graph.edges[keys.EDGE_LENGTHS]
    lengths_arr = lengths.array if isinstance(lengths, e3j.IrrepsArray) else lengths
    np.testing.assert_allclose(np.asarray(lengths_arr), [[1.0], [1.0]])


def test_with_edge_vectors_masked_edges():
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    graph = gcnn.graph_from_points(pos, r_max=2)
    graph = gcnn.data.pad_with_graphs(graph, n_node=3, n_edge=3)
    graph = gcnn.data.add_padding_mask(graph, overwrite=True)
    # zero out one of the two real edges
    graph.edges[keys.MASK] = np.asarray(graph.edges[keys.MASK]).copy()
    graph.edges[keys.MASK][0] = False

    result = gcnn.with_edge_vectors(graph)

    assert jnp.all(result.edges[keys.EDGE_VECTORS].array[0] == 1.0)
    assert result.edges[keys.EDGE_LENGTHS].array[0] == 0.0


@pytest.mark.parametrize("jit", (False, True))
def test_with_edge_vectors_grad(jit):
    length = 1.0
    pos = jnp.array([[0.0, 0.0, 0.0], [length, 0.0, 0.0]])
    graph = gcnn.graph_from_points(pos, r_max=2.0)
    graph = jraph.pad_with_graphs(graph, n_node=len(pos) + 2, n_edge=len(pos) + 2, n_graph=2)
    graph = gcnn.data.add_padding_mask(graph)

    def get_length(graph, pos: jax.Array):
        graph.nodes[gcnn.keys.POSITIONS] = pos
        graph = gcnn.with_edge_vectors(graph)

        n_graph = graph.n_edge.shape[0]
        graph_idx = jnp.arange(n_graph)
        sum_n_edge = jax.tree_util.tree_leaves(graph.edges)[0].shape[0]
        node_gr_idx = jnp.repeat(graph_idx, graph.n_edge, axis=0, total_repeat_length=sum_n_edge)

        inputs = graph.edges[gcnn.keys.EDGE_LENGTHS].array
        return jnp.sum(
            jax.tree_util.tree_map(lambda n: jraph.segment_sum(n, node_gr_idx, n_graph), inputs)
        )

    get_length = functools.partial(get_length, graph)
    if jit:
        get_length = jax.jit(get_length)

    length_, grad = jax.value_and_grad(get_length)(pos)
    # Two times length as we sum the length from 0->1 and 1->0
    assert jnp.isclose(length_, 2 * length)
    assert jnp.array_equal(grad, jnp.array([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))


def test_pairwise_sq_distances():
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    expected = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=float)
    np.testing.assert_allclose(_spatial._pairwise_sq_distances(pos, None), expected)


def test_pairwise_sq_distances_masked():
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    mask = np.array([True, False, True])
    result = np.asarray(_spatial._pairwise_sq_distances(pos, mask))

    # the masked node's row and column are pushed to inf (or nan for inf - inf)
    assert np.isinf(result[1, [0, 2]]).all()
    assert not np.isfinite(result[1, 1]).all()
    assert np.all(result[np.ix_([0, 2], [0, 2])] == np.array([[0, 2], [2, 0]]))


def test_update_positions():
    pos = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [10, 10, 10]])
    graph = gcnn.graph_from_points(pos, r_max=8.0)
    graph = gcnn.data.pad_with_graphs(graph, n_node=len(pos) + 2, n_edge=len(graph.edges) + 10)
    graph = gcnn.data.add_padding_mask(graph, overwrite=True)

    new_pos = jnp.array(
        [[0.0, 0.0, 0.0], [0.6, 0.0, 0.0], [10.0, 10.0, 10.0], [0, 0, 0], [0, 0, 0]]
    )
    result = _spatial._update_positions(graph, new_pos, r_max=1.0)

    pairs = set(zip(result.senders.tolist(), result.receivers.tolist()))
    assert (0, 1) in pairs
    assert (1, 0) in pairs

    n_edge = result.edges[keys.MASK].shape[0]
    n_valid = int(result.n_edge[0])
    assert n_valid == 2
    assert n_edge == n_valid + result.n_edge[1]
    np.testing.assert_allclose(np.asarray(result.nodes[keys.POSITIONS]), np.asarray(new_pos))


def test_update_positions_shape_mismatch():
    pos = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    graph = gcnn.graph_from_points(pos, r_max=1.0)
    graph = gcnn.data.pad_with_graphs(graph, n_node=len(pos) + 1, n_edge=len(graph.edges) + 4)
    graph = gcnn.data.add_padding_mask(graph, overwrite=True)

    with pytest.raises(ValueError, match="same shape"):
        _spatial._update_positions(graph, jnp.ones((2, 3)), r_max=1.0)
