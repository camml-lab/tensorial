from typing import Literal

import e3nn_jax as e3j
import jax
import jax.numpy as jnp
import jaxtyping as jt
import jraph
from pytray import tree

from . import keys, utils
from .. import nn_utils

__all__ = (
    "segment_sum",
    "segment_mean",
    "segment_max",
    "segment_min",
    "segment_reduce",
    "graph_segment_reduce",
)


def _prepare_segments(segment_sizes, total_repeat_length: int):
    num_segments: int = segment_sizes.shape[0]

    # 1. Generate segment IDs (map each data point to its graph index)
    segment_ids: jt.Int[jax.Array, "num_segments"] = jnp.arange(num_segments)
    # total_repeat_length ensures correct size even with padding/dynamic shapes
    segment_ids = jnp.repeat(
        segment_ids, segment_sizes, axis=0, total_repeat_length=total_repeat_length
    )

    return num_segments, segment_ids


def segment_sum(
    data: jt.Float[jax.Array, "N ..."],
    segment_sizes: jt.Int[jax.Array, "num_segments"],
    mask: jt.Bool[jax.Array, "N ..."] | None = None,
) -> jax.Array:
    """
    Performs a masked segment reduction (sum or mean) over batched graph data.

    This function is JAX-jittable and handles the logic for applying a mask
    before reduction, ensuring correct gradient flow and shape consistency.

    Args:
        data: The array of values to reduce (e.g., loss, features). Shape (N_total, D) or
            (N_total,).
        segment_sizes: The array of segment sizes (e.g., jraph.GraphsTuple.n_node).
                          Shape (num_segments,).
        reduction: The type of reduction to perform ("sum" or "mean").
        mask: Optional boolean array indicating valid entries. Shape (N_total,).

    Returns:
        The reduced array (segment sum or mean). Shape (num_segments, D) or (num_segments,).
    """
    num_segments, segment_ids = _prepare_segments(segment_sizes, data.shape[0])

    # 2. Prepare Masked Numerator (Sum)
    if mask is not None:
        # Apply the mask by multiplication (zeros out invalid data).
        # Ensure mask can broadcast to data (e.g., (N,) -> (N, 1) for (N, D) data).
        mask = nn_utils.prepare_mask(mask, data)
        data = data * mask

    # Segment Sum of the masked data (Numerator for the mean)
    return _jraph_segment(
        data,
        segment_ids=segment_ids,
        num_segments=num_segments,
        indices_are_sorted=True,
        reduction="sum",
    )


def segment_mean(
    data: jt.Float[jax.Array, "N ..."],
    segment_sizes: jt.Int[jax.Array, "num_segments"],
    mask: jt.Bool[jax.Array, "N ..."] | None = None,
) -> jax.Array:
    """
    Performs a masked segment mean reduction over batched graph data.

    Args:
        data: The array of values to reduce (e.g., features). Shape (N_total, D) or
            (N_total,).
        segment_sizes: The array of segment sizes (e.g., jraph.GraphsTuple.n_node).
                          Shape (num_segments,).
        mask: Optional boolean array indicating valid entries. Shape (N_total,).

    Returns:
        The reduced array (segment mean). Shape (num_segments, D) or (num_segments,).
    """
    num_segments, segment_ids = _prepare_segments(segment_sizes, data.shape[0])

    # 2. Prepare Masked Numerator (Sum)
    if mask is None:
        # If no mask is provided, treat all entries as valid (mask = 1)
        mask_int = jnp.ones(data.shape[0], dtype=jnp.int32)
    else:
        mask_int = mask.astype(jnp.int32)

        # Apply the mask by multiplication (zeros out invalid data).
        # Ensure mask can broadcast to data (e.g., (N,) -> (N, 1) for (N, D) data).
        mask = nn_utils.prepare_mask(mask, data)
        data = data * mask

    # Segment Sum of the masked data (Numerator for the mean)
    data_sum = _jraph_segment(
        data,
        segment_ids=segment_ids,
        num_segments=num_segments,
        indices_are_sorted=True,
        reduction="sum",
    )

    # 3. Handle Reduction Type
    # Segment Sum of the mask (Denominator for the mean - the count)
    count_data_sum = jraph.segment_sum(
        data=mask_int,
        segment_ids=segment_ids,
        num_segments=num_segments,
        indices_are_sorted=True,
    )

    # Prepare count for broadcast division (B, 1) or (B,)
    safe_counts = count_data_sum
    if data_sum.ndim > count_data_sum.ndim:
        safe_counts = count_data_sum[:, None]

    # Calculate mean using jnp.where for robustness and jittability:
    # If count > 0, calculate mean; otherwise, return 0.
    return jnp.where(safe_counts > 0, data_sum / safe_counts, jnp.zeros_like(data_sum))


def segment_min(
    data: jt.Float[jax.Array, "N ..."],
    segment_sizes: jt.Int[jax.Array, "num_segments"],
    mask: jt.Bool[jax.Array, "N ..."] | None = None,
) -> jax.Array:
    """
    Performs a masked segment minimum reduction over batched graph data.

    Args:
        data: The array of values to reduce (e.g., features). Shape (N_total, D) or
            (N_total,).
        segment_sizes: The array of segment sizes (e.g., jraph.GraphsTuple.n_node).
                          Shape (num_segments,).
        mask: Optional boolean array indicating valid entries. Shape (N_total,).

    Returns:
        The reduced array (segment minimum). Shape (num_segments, D) or (num_segments,).
    """
    num_segments, segment_ids = _prepare_segments(segment_sizes, data.shape[0])

    # 1. Handle Masking for MIN operation
    if mask is not None:
        # Prepare mask for broadcast (e.g., (N,) -> (N, 1))
        prepared_mask = nn_utils.prepare_mask(mask, data)

        # Set invalid (masked-out) values to POSITIVE INFINITY.
        # This ensures they are ignored when finding the minimum.
        data = jnp.where(prepared_mask, data, jnp.full(data.shape, jnp.inf, dtype=data.dtype))

    # 2. Perform Segment Minimum Reduction
    data_min = _jraph_segment(
        data,
        segment_ids=segment_ids,
        num_segments=num_segments,
        indices_are_sorted=True,
        reduction="min",
    )

    # 3. Handle Empty/Fully-Masked Segments
    # Segments that were empty or fully masked will result in Inf.
    # We replace these with 0 (or your chosen default for an empty segment).
    is_inf = jnp.isinf(data_min)

    # Replace Inf with 0, otherwise keep the minimum value found.
    segment_min_safe = jnp.where(is_inf, jnp.zeros_like(data_min), data_min)

    return segment_min_safe


def segment_max(
    data: jt.Float[jax.Array, "N ..."],
    segment_sizes: jt.Int[jax.Array, "num_segments"],
    mask: jt.Bool[jax.Array, "N ..."] | None = None,
) -> jax.Array:
    """
    Performs a masked segment maximum reduction over batched graph data.

    Args:
        data: The array of values to reduce (e.g., features). Shape (N_total, D) or
            (N_total,).
        segment_sizes: The array of segment sizes (e.g., jraph.GraphsTuple.n_node).
                          Shape (num_segments,).
        mask: Optional boolean array indicating valid entries. Shape (N_total,).

    Returns:
        The reduced array (segment maximum). Shape (num_segments, D) or (num_segments,).
    """
    num_segments, segment_ids = _prepare_segments(segment_sizes, data.shape[0])

    # 1. Handle Masking for MAX operation
    if mask is not None:
        # Prepare mask for broadcast (e.g., (N,) -> (N, 1))
        prepared_mask = nn_utils.prepare_mask(mask, data)

        # Set invalid (masked-out) values to POSITIVE INFINITY.
        # This ensures they are ignored when finding the maximum.
        data = jnp.where(prepared_mask, data, jnp.full(data.shape, -jnp.inf, dtype=data.dtype))

    # 2. Perform Segment maximum Reduction
    data_max = _jraph_segment(
        data,
        segment_ids=segment_ids,
        num_segments=num_segments,
        indices_are_sorted=True,
        reduction="max",
    )

    # 3. Handle Empty/Fully-Masked Segments
    # Segments that were empty or fully masked will result in Inf.
    # We replace these with 0 (or your chosen default for an empty segment).
    is_inf = jnp.isinf(data_max)

    # Replace Inf with 0, otherwise keep the maximum value found.
    segment_max_safe = jnp.where(is_inf, jnp.zeros_like(data_max), data_max)

    return segment_max_safe


_REDUCTIONS = {
    "mean": segment_mean,
    "sum": segment_sum,
    "min": segment_min,
    "max": segment_max,
}


def segment_reduce(
    data: jt.Float[jax.Array | e3j.IrrepsArray, "N ..."],
    segment_sizes: jt.Int[jax.Array, "num_segments"],
    reduction: Literal["sum", "mean"],
    mask: jt.Bool[jax.Array, "N ..."] | None = None,
) -> jax.Array:
    """
    Performs a masked segment reduction (sum or mean) over batched graph data.

    This function is JAX-jittable and handles the logic for applying a mask
    before reduction, ensuring correct gradient flow and shape consistency.

    Args:
        data: The array of values to reduce (e.g., loss, features). Shape (N_total, D) or
            (N_total,).
        segment_sizes: The array of segment sizes (e.g., jraph.GraphsTuple.n_node).
                          Shape (num_segments,).
        reduction: The type of reduction to perform ("sum" or "mean").
        mask: Optional boolean array indicating valid entries. Shape (N_total,).

    Returns:
        The reduced array (segment sum or mean). Shape (num_segments, D) or (num_segments,).
    """
    # 3Handle Reduction Type
    try:
        fn = _REDUCTIONS[reduction]
        return fn(data, segment_sizes, mask=mask)
    except KeyError:
        # Raise an error using JAX's preferred method for errors in jitted regions
        # (Though, generally better to handle non-jittable logic outside the jit block)
        raise ValueError(
            f"Unsupported reduction type: {reduction}. "
            f"Must be one of {list(_REDUCTIONS.keys())}."
        ) from None


def graph_segment_reduce(
    graph: jraph.GraphsTuple | dict,
    path: "gcnn.typing.TreePathLike",
    reduction: str = "sum",
) -> e3j.IrrepsArray | jax.Array:
    if isinstance(graph, jraph.GraphsTuple):
        graph_dict = graph._asdict()
    else:
        graph_dict = graph

    path = utils.path_from_str(path)
    root = path[0]
    if root == "nodes":
        n_type = graph_dict["n_node"]
    elif root == "edges":
        n_type = graph_dict["n_edge"]
    else:
        raise ValueError(f"Reduce can only act on nodes or edges, got {path}")

    try:
        inputs = tree.get_by_path(graph_dict, path)
    except KeyError:
        raise ValueError(f"Could not find field '{path}' in graph") from None

    mask = graph_dict[root].get(keys.MASK)
    return segment_reduce(inputs, n_type, reduction=reduction, mask=mask)


def _jraph_segment(
    inputs: e3j.IrrepsArray | jax.Array,
    segment_ids: jax.Array,
    num_segments: int | None = None,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    reduction: str = "sum",
) -> e3j.IrrepsArray | jax.Array:
    try:
        op = getattr(jraph, f"segment_{reduction}")
    except AttributeError:
        raise ValueError(f"Unknown reduction operation: {reduction}") from None

    return jax.tree_util.tree_map(
        lambda n: op(n, segment_ids, num_segments, indices_are_sorted, unique_indices), inputs
    )
