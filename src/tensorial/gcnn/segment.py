from typing import Literal

import jax
import jax.numpy as jnp
import jaxtyping as jt
import jraph

from .. import nn_utils


def reduce_masked(
    data: jt.Float[jax.Array, "N ..."],
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
    num_segments: int = segment_sizes.shape[0]

    # 1. Generate segment IDs (map each data point to its graph index)
    graph_idx: jt.Int[jax.Array, "num_segments"] = jnp.arange(num_segments)
    # total_repeat_length ensures correct size even with padding/dynamic shapes
    segment_ids = jnp.repeat(graph_idx, segment_sizes, axis=0, total_repeat_length=data.shape[0])

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
    data_sum = jraph.segment_sum(
        data=data, segment_ids=segment_ids, num_segments=num_segments, indices_are_sorted=True
    )

    # 3. Handle Reduction Type
    if reduction == "sum":
        return data_sum

    if reduction == "mean":
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
        segment_mean = jnp.where(safe_counts > 0, data_sum / safe_counts, jnp.zeros_like(data_sum))
        return segment_mean

    # Raise an error using JAX's preferred method for errors in jitted regions
    # (Though, generally better to handle non-jittable logic outside the jit block)
    raise ValueError(f"Unsupported reduction type: {reduction}. Must be 'sum' or 'mean'.")
