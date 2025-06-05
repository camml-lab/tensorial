"""
Common utility functions that operate on graphs
"""

from typing import TYPE_CHECKING, Optional, Union

import beartype
import e3nn_jax as e3j
import jax
import jax.numpy as jnp
import jaxtyping as jt
import jraph
from pytray import tree

from . import utils

if TYPE_CHECKING:
    import tensorial
    from tensorial import gcnn


__all__ = ("reduce",)


@jt.jaxtyped(typechecker=beartype.beartype)
def reduce(
    graph: Union[jraph.GraphsTuple, dict],
    field: "gcnn.typing.TreePathLike",
    reduction: str = "sum",
) -> Union[e3j.IrrepsArray, jax.Array]:
    if isinstance(graph, (jraph.GraphsTuple)):
        graph_dict = graph._asdict()
    else:
        graph_dict = graph

    field = utils.path_from_str(field)
    if field[0] == "nodes":
        n_type = graph_dict["n_node"]
    elif field[0] == "edges":
        n_type = graph["n_edge"]
    else:
        raise ValueError(f"Reduce can only act on nodes or edges, got {field}")

    try:
        inputs = tree.get_by_path(graph_dict, field)
    except KeyError:
        raise ValueError(f"Could not find field '{field}' in graph") from None

    num_segments = jax.tree_util.tree_leaves(graph_dict[field[0]])[0].shape[0]

    return _reduce(
        inputs=inputs, segment_lengths=n_type, num_segments=num_segments, reduction=reduction
    )


@jt.jaxtyped(typechecker=beartype.beartype)
def _reduce(
    inputs: "tensorial.typing.ArrayType",
    segment_lengths: "tensorial.typing.ArrayType",
    num_segments: Optional[int] = None,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    reduction: str = "sum",
) -> Union[e3j.IrrepsArray, jax.Array]:
    try:
        op = getattr(jraph, f"segment_{reduction}")
    except AttributeError:
        raise ValueError(f"Unknown reduction operation: {reduction}") from None

    # this aggregation follows jraph/_src/models.py
    n_graph = segment_lengths.shape[0]
    graph_idx = jnp.arange(n_graph)
    node_gr_idx = jnp.repeat(graph_idx, segment_lengths, axis=0, total_repeat_length=num_segments)

    return jax.tree_util.tree_map(
        lambda n: op(n, node_gr_idx, n_graph, indices_are_sorted, unique_indices), inputs
    )
