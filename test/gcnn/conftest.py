# -*- coding: utf-8 -*-
import random

import e3nn_jax as e3j
import jax
import jax.numpy as jnp
import jraph
import pytest

from tensorial import gcnn
from tensorial.gcnn import keys


@pytest.fixture
def cube_graph() -> jraph.GraphsTuple:
    # Create coordinates for the corners of a cube
    coords = (-1.0, 1.0)
    pts = []
    for i in coords:
        for j in coords:
            for k in coords:
                pts.append([i, j, k])

    pts = jnp.array(pts)
    node_species = jnp.array(random.choices([0, 1, 2], k=len(pts)))
    nodes = {
        gcnn.keys.SPECIES: node_species,
        gcnn.keys.ATTRIBUTES: e3j.as_irreps_array(jax.nn.one_hot(node_species, len(pts))),
    }

    graph = gcnn.graph_from_points(pts, r_max=2.0, nodes=nodes)  # We don't connect across diagonals
    assert graph.n_node[0] == 8
    # Graph contains two edges for along each edge of the cube i.e. i->j, j->i
    assert graph.n_edge[0] == 24
    return gcnn.with_edge_vectors(graph)


@pytest.fixture
def cube_graph_gcnn(cube_graph):  # pylint: disable=redefined-outer-name
    nodes = cube_graph.nodes
    nodes[keys.FEATURES] = nodes[keys.ATTRIBUTES]
    edges = cube_graph.edges
    edges[keys.RADIAL_EMBEDDINGS] = e3j.bessel(edges[keys.EDGE_LENGTHS][:, 0], 4, 2.0)
    edges[keys.ATTRIBUTES] = e3j.spherical_harmonics(
        "1o + 2e",
        edges[keys.EDGE_VECTORS],
        normalize=True,
    )
    return cube_graph._replace(nodes=nodes, edges=edges)
