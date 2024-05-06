# -*- coding: utf-8 -*-
import jax
from jax import random
import jax.numpy as jnp
import jraph

from tensorial import gcnn
from tensorial.gcnn import keys


def test_grads():
    def get_norms(pos):
        graph = gcnn.with_edge_vectors(gcnn.graph_from_points(pos, r_max=2.0))
        return graph.edges[keys.EDGE_LENGTHS][0, 0]

    pos = jnp.array(
        [
            [
                0.0,
                0.0,
                0.0,
            ],
            [
                1.0,
                1.0,
                1.0,
            ],
        ]
    )

    norms = get_norms(pos)
    assert jnp.allclose(norms, jnp.sqrt(3.0))

    grads = jax.grad(get_norms)(pos)
    assert jnp.allclose(jnp.abs(grads), 1.0 / jnp.sqrt(3.0))


def test_grad_module():
    def get_energy(g):
        g = gcnn.with_edge_vectors(g)
        energy = g.edges[keys.EDGE_LENGTHS] ** 2
        _per_atom_energy = jraph.segment_sum(
            energy, graph.receivers, num_segments=len(graph.nodes[keys.POSITIONS])
        )
        return g._replace(globals=dict(total_energy=energy))

    pos = jnp.array(
        [
            [
                0.0,
                0.0,
                0.0,
            ],
            [
                1.0,
                1.0,
                1.0,
            ],
        ]
    )
    graph = gcnn.graph_from_points(pos, r_max=2.0)
    graph2 = gcnn.graph_from_points(pos, r_max=2.0)
    graph = jraph.batch([graph, graph2])

    grad = gcnn.Grad(
        get_energy,
        "globals.total_energy",
        wrt="nodes.positions",
    )

    params = grad.init(random.key(0), graph)
    _res = grad.apply(params, graph)

    _res = jax.jit(grad.apply)(params, graph)  # pylint: disable=not-callable
