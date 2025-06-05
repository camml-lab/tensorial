import functools

import jax
import jax.numpy as jnp
import jraph

import tensorial
from tensorial import gcnn
from tensorial.gcnn import keys


def test_grads():
    def get_norms(pos, graph):
        # Have to do a strange thing here where we set the positions (even though they
        # already exist) to make this function a function of the positions that we can then
        # take derivatives of
        graph.nodes[keys.POSITIONS] = pos
        graph = gcnn.with_edge_vectors(graph)
        return tensorial.as_array(graph.edges[keys.EDGE_LENGTHS])[0, 0]

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
    assert jnp.allclose(jnp.abs(res.edges[f"denergy/d{keys.EDGE_VECTORS}"]), 2.0)
