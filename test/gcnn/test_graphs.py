import functools

import jax
import jax.numpy as jnp
import jraph
import numpy as np
import pytest

from tensorial import gcnn


def test_graph_from_points():
    # Check that 1D
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    gcnn.graph_from_points(pos, r_max=2)

    # and 2D work
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    gcnn.graph_from_points(pos, r_max=2)


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
