"""
Common utility functions that operate on graphs
"""

from typing import Union

import jax
import jax.numpy as jnp
import jraph
from pytray import tree

from . import utils

__all__ = ("reduce",)


def reduce(graph: jraph.GraphsTuple, field: Union[str, jax.Array], reduction="sum") -> jax.Array:
    try:
        op = getattr(jraph, f"segment_{reduction}")
    except AttributeError:
        raise ValueError(f"Unknown reduction operation: {reduction}") from None

    graph_dict = graph._asdict()
    field = utils.path_from_str(field)
    if field[0] == "nodes":
        n_type = graph.n_node
    elif field[0] == "edges":
        n_type = graph.n_edge
    else:
        raise ValueError(f"Reduce can only act on nodes or edges, got {field}")

    # this aggregation follows jraph/_src/models.py
    n_graph = n_type.shape[0]
    graph_idx = jnp.arange(n_graph)
    sum_n_type = jax.tree_util.tree_leaves(graph_dict[field[0]])[0].shape[0]
    node_gr_idx = jnp.repeat(graph_idx, n_type, axis=0, total_repeat_length=sum_n_type)

    inputs = tree.get_by_path(graph_dict, field)
    return jax.tree_util.tree_map(lambda n: op(n, node_gr_idx, n_graph), inputs)
