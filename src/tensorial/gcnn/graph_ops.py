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


def _prepare_segments(
    segment_sizes: jt.Int[jax.Array, "num_segments"], total_repeat_length: int
) -> tuple[int, jt.Int[jax.Array, "total_repeat_length"]]:
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
    segment_mask: jt.Bool[jax.Array, "num_segments"] | None = None,
) -> jt.Float[jax.Array, "num_segments ..."]:
    """Performs a masked segment sum reduction over batched graph data.

    This function is JAX-jittable and handles the logic for applying a mask
    before reduction, ensuring correct gradient flow and shape consistency.

    Args:
        data: The array of values to reduce (e.g., loss, features). Shape (N_total, D) or
            (N_total,).
        segment_sizes: The array of segment sizes (e.g., jraph.GraphsTuple.n_node).
            Shape (num_segments,).
        mask: Optional boolean array indicating valid entries. Shape (N_total,).
        segment_mask: Optional boolean array indicating valid segments. Shape (num_segments,).
            If None, no additional segment-level invalidation is applied; empty/fully-masked
            segments naturally yield 0.

    Returns:
        The reduced array (segment sum). Shape (num_segments, D) or (num_segments,).
        Invalid segments (where segment_mask is False) will have value 0.
    """
    num_segments, segment_ids = _prepare_segments(segment_sizes, data.shape[0])

    # 2. Prepare Masked Numerator (Sum)
    if mask is not None:
        # Apply the mask by multiplication (zeros out invalid data).
        # Ensure mask can broadcast to data (e.g., (N,) -> (N, 1) for (N, D) data).
        mask = nn_utils.prepare_mask(mask, data)
        data = data * mask

    # Segment Sum of the masked data (Numerator for the mean)
    result = _jraph_segment(
        data,
        segment_ids=segment_ids,
        num_segments=num_segments,
        indices_are_sorted=True,
        reduction="sum",
    )

    if segment_mask is not None:
        segment_mask = nn_utils.prepare_mask(segment_mask, result)
        result = jnp.where(segment_mask, result, jnp.zeros_like(result))

    return result


def segment_mean(
    data: jt.Float[jax.Array, "N ..."],
    segment_sizes: jt.Int[jax.Array, "num_segments"],
    mask: jt.Bool[jax.Array, "N ..."] | None = None,
    segment_mask: jt.Bool[jax.Array, "num_segments"] | None = None,
) -> jt.Float[jax.Array, "num_segments ..."]:
    """Performs a masked segment mean reduction over batched graph data.

    Args:
        data: The array of values to reduce (e.g., features). Shape (N_total, D) or
            (N_total,).
        segment_sizes: The array of segment sizes (e.g., jraph.GraphsTuple.n_node).
            Shape (num_segments,).
        mask: Optional boolean array indicating valid entries. Shape (N_total,).
        segment_mask: Optional boolean array indicating valid segments. Shape (num_segments,).
            If None, it is inferred from data counts (segments with >0 valid items are valid).

    Returns:
        The reduced array (segment mean). Shape (num_segments, D) or (num_segments,).
    """
    num_segments, segment_ids = _prepare_segments(segment_sizes, data.shape[0])

    # 1. Prepare Masked Numerator (Sum)
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

    # 2. Handle Reduction Type
    # Segment Sum of the mask (Denominator for the mean - the count)
    count_data_sum = jraph.segment_sum(
        data=mask_int,
        segment_ids=segment_ids,
        num_segments=num_segments,
        indices_are_sorted=True,
    )

    # 3. Handle masked segments
    # Prepare count for broadcast division (B, 1) or (B,)
    safe_counts = count_data_sum
    if data_sum.ndim > count_data_sum.ndim:
        safe_counts = count_data_sum[:, None]

    # If count > 0, calculate mean; otherwise, return 0.
    if segment_mask is None:
        segment_mask = safe_counts != 0
    else:
        # Ensure segment_mask broadcasts if safe_counts has extra dims
        segment_mask = nn_utils.prepare_mask(segment_mask, safe_counts)

    mean = jnp.where(
        safe_counts > 0,
        data_sum / jnp.where(segment_mask, safe_counts, 1.0),
        jnp.zeros_like(data_sum),
    )
    return jnp.where(segment_mask, mean, jnp.zeros_like(mean))


def segment_min(
    data: jt.Float[jax.Array, "N ..."],
    segment_sizes: jt.Int[jax.Array, "num_segments"],
    mask: jt.Bool[jax.Array, "N ..."] | None = None,
    segment_mask: jt.Bool[jax.Array, "num_segments"] | None = None,
) -> jt.Float[jax.Array, "num_segments ..."]:
    """Performs a masked segment minimum reduction over batched graph data.

    Args:
        data: The array of values to reduce (e.g., features). Shape (N_total, D) or
            (N_total,).
        segment_sizes: The array of segment sizes (e.g., jraph.GraphsTuple.n_node).
            Shape (num_segments,).
        mask: Optional boolean array indicating valid entries. Shape (N_total,).
        segment_mask: Optional boolean array indicating valid segments. Shape (num_segments,).
            If None, no additional segment invalidation is applied; empty/fully-masked
            segments naturally yield +inf.

    Returns:
        The reduced array (segment minimum). Shape (num_segments, D) or (num_segments,).
        Invalid segments (where segment_mask is False) will have value +inf.
    """
    num_segments, segment_ids = _prepare_segments(segment_sizes, data.shape[0])

    # 1. Handle Masking for MIN operation
    if mask is not None:
        # Prepare mask for broadcast (e.g., (N,) -> (N, 1))
        mask = nn_utils.prepare_mask(mask, data)

        # Set invalid (masked-out) values to POSITIVE INFINITY.
        # This ensures they are ignored when finding the minimum.
        data = jnp.where(mask, data, jnp.inf)

    # 2. Perform Segment Minimum Reduction
    data_min = _jraph_segment(
        data,
        segment_ids=segment_ids,
        num_segments=num_segments,
        indices_are_sorted=True,
        reduction="min",
    )

    # 3. Handle Empty/Fully-Masked Segments
    # If segment_mask is provided, invalid segments should be +inf.
    if segment_mask is not None:
        # Broadcast segment_mask
        segment_mask = nn_utils.prepare_mask(segment_mask, data_min)
        data_min = jnp.where(segment_mask, data_min, jnp.inf)

    return data_min


def segment_max(
    data: jt.Float[jax.Array, "N ..."],
    segment_sizes: jt.Int[jax.Array, "num_segments"],
    mask: jt.Bool[jax.Array, "N ..."] | None = None,
    segment_mask: jt.Bool[jax.Array, "num_segments"] | None = None,
) -> jt.Float[jax.Array, "num_segments ..."]:
    """Performs a masked segment maximum reduction over batched graph data.

    Args:
        data: The array of values to reduce (e.g., features). Shape (N_total, D) or
            (N_total,).
        segment_sizes: The array of segment sizes (e.g., jraph.GraphsTuple.n_node).
            Shape (num_segments,).
        mask: Optional boolean array indicating valid entries. Shape (N_total,).
        segment_mask: Optional boolean array indicating valid segments. Shape (num_segments,).
            If None, no additional segment invalidation is applied; empty/fully-masked
            segments naturally yield -inf.

    Returns:
        The reduced array (segment maximum). Shape (num_segments, D) or (num_segments,).
        Invalid segments (where segment_mask is False) will have value -inf.
    """
    num_segments, segment_ids = _prepare_segments(segment_sizes, data.shape[0])

    # 1. Handle Masking for MAX operation
    if mask is not None:
        # Prepare mask for broadcast (e.g., (N,) -> (N, 1))
        mask = nn_utils.prepare_mask(mask, data)

        # Set invalid (masked-out) values to NEGATIVE INFINITY.
        # This ensures they are ignored when finding the maximum.
        data = jnp.where(mask, data, -jnp.inf)

    # 2. Perform Segment maximum Reduction
    data_max = _jraph_segment(
        data,
        segment_ids=segment_ids,
        num_segments=num_segments,
        indices_are_sorted=True,
        reduction="max",
    )

    # 3. Handle Empty/Fully-Masked Segments
    # If segment_mask is provided, invalid segments should be -inf.
    if segment_mask is not None:
        # Broadcast segment_mask
        segment_mask = nn_utils.prepare_mask(segment_mask, data_max)
        data_max = jnp.where(segment_mask, data_max, -jnp.inf)

    return data_max


_REDUCTIONS = {
    "mean": segment_mean,
    "sum": segment_sum,
    "min": segment_min,
    "max": segment_max,
}


def segment_reduce(
    data: jt.Float[jax.Array | e3j.IrrepsArray, "N ..."],
    segment_sizes: jt.Int[jax.Array, "num_segments"],
    reduction: Literal["sum", "mean", "min", "max"],
    mask: jt.Bool[jax.Array, "N ..."] | None = None,
    segment_mask: jt.Bool[jax.Array, "num_segments"] | None = None,
) -> jt.Float[jax.Array, "num_segments ..."]:
    """Performs a masked segment reduction over batched graph data.

    This function is JAX-jittable and handles the logic for applying a mask
    before reduction, ensuring correct gradient flow and shape consistency.

    Args:
        data: The array of values to reduce (e.g., loss, features). Shape (N_total, D) or
            (N_total,).
        segment_sizes: The array of segment sizes (e.g., jraph.GraphsTuple.n_node).
            Shape (num_segments,).
        reduction: The type of reduction to perform ("sum", "mean", "min", "max").
        mask: Optional boolean array indicating valid entries. Shape (N_total,).
        segment_mask: Optional boolean array indicating valid segments. Shape (num_segments,).

    Returns:
        The reduced array. Shape (num_segments, D) or (num_segments,).
    """
    # 3Handle Reduction Type
    try:
        fn = _REDUCTIONS[reduction]
        return fn(data, segment_sizes, mask=mask, segment_mask=segment_mask)
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
) -> jt.Float[jax.Array, "num_segments ..."] | e3j.IrrepsArray:
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
) -> jt.Float[jax.Array, "num_segments ..."] | e3j.IrrepsArray:
    try:
        op = getattr(jraph, f"segment_{reduction}")
    except AttributeError:
        raise ValueError(f"Unknown reduction operation: {reduction}") from None

    return jax.tree_util.tree_map(
        lambda n: op(n, segment_ids, num_segments, indices_are_sorted, unique_indices), inputs
    )
