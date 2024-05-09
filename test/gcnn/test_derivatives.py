# -*- coding: utf-8 -*-
import jax
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


def test_grad_module(rng_key):
    def get_energy(graph_):
        graph_ = gcnn.with_edge_vectors(graph_)
        return graph_

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
        of=f"edges.{keys.EDGE_LENGTHS}",
        wrt="nodes.positions",
    )

    params = grad.init(rng_key, graph)
    res = grad.apply(params, graph)
    assert jnp.allclose(jnp.abs(res.edges[f"d{keys.EDGE_LENGTHS}/dpositions"]), 2.0 / jnp.sqrt(3.0))
