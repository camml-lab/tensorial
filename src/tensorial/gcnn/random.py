# -*- coding: utf-8 -*-
from __future__ import annotations  # For py39

from typing import Sequence

import jax.random
import jraph

from . import _graphs


def spatial_graph(
    key: jax.Array, num_nodes: int = None, num_graphs=None, cutoff=0.2
) -> jraph.GraphsTuple | Sequence[jraph.GraphsTuple]:
    graphs = []
    for _ in range(num_graphs or 1):
        if num_nodes is None:
            new_key, key = jax.random.split(key)
            num_nodes = jax.random.randint(new_key, shape=(), minval=2, maxval=10)

        new_key, key = jax.random.split(key)
        pos = jax.random.uniform(new_key, shape=(num_nodes, 3))

        graphs.append(_graphs.graph_from_points(pos, r_max=cutoff))

    if num_graphs is None:
        return graphs[0]

    return graphs
