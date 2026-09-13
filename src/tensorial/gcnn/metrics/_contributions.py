from collections.abc import Sequence
from typing import Literal

import beartype
import jax
import jax.numpy as jnp
import jaxtyping as jt
from jaxtyping import Array, Bool, Float, Int
import jraph
from pytray import tree
import reax
from typing_extensions import override

from .. import _tree, graph_ops, keys, typing
from ... import nn_utils, utils

__all__ = "TypeContributionLstsq", "PropertyContributionLstsq"


@beartype.beartype
class TypeContributionLstsq(reax.metrics.Metric[Array]):
    """Online Least Squares Metric.

    Uses 'Sufficient Statistics' (XtX, Xty) to perform linear regression
    without storing the entire dataset history.
    """

    # XtX: The Gram Matrix (A.T @ A) -> Shape: (n_types, n_types)
    xtx: Float[jax.Array, "n_types n_types"] | None = None

    # Xty: The Moment Vector (A.T @ b) -> Shape: (n_types, ...)
    xty: Float[jax.Array, "n_types ..."] | None = None

    @property
    def is_empty(self):
        return self.xtx is None

    @classmethod
    @override
    def empty(cls) -> "TypeContributionLstsq":  # pylint: disable=arguments-differ
        return cls()

    @classmethod
    @override
    def create(  # pylint: disable=arguments-differ
        cls,
        type_counts: Int[Array, "batch_size n_types"] | Float[Array, "batch_size n_types"],
        values: Float[Array, "batch_size ..."],
        mask: Bool[Array, "batch_size"] | None = None,
        /,
    ) -> "TypeContributionLstsq":
        np_ = utils.infer_backend(type_counts)

        # 1. Cast inputs to float for matrix operations
        a_mtx = type_counts.astype(np_.float32)
        b_vec = values

        # 2. Apply shape-stable masking
        # Instead of A[mask] (which changes shape), we use jnp.where.
        if mask is not None:
            # Broadcast mask: (batch,) -> (batch, 1)
            mask_expanded = mask[:, None]

            # CRITICAL: Use jnp.where instead of (A * mask).
            # If the masked-out rows in A contain NaNs, (NaN * 0) is still NaN.
            # jnp.where ensures safe zeros are used for ignored rows.
            a_mtx = jnp.where(mask_expanded, a_mtx, 0.0)

            # Handle masking for 'b' (values)
            # We align the mask dimensions to match b
            if b_vec.ndim > 1:
                # If b is (batch, targets), reshape mask to (batch, 1)
                mask_b = mask.reshape((mask.shape[0],) + (1,) * (b_vec.ndim - 1))
            else:
                # If b is (batch,), standard mask works
                mask_b = mask

            b_vec = jnp.where(mask_b, b_vec, 0.0)

        # 3. Compute Sufficient Statistics for this batch
        # Since ignored rows are now exactly 0.0, they add nothing to the result
        # of the matrix multiplication, effectively filtering them out.

        # A.T @ A -> (n_types, n_types)
        batch_xtx = a_mtx.T @ a_mtx

        # A.T @ b -> (n_types, ...)
        batch_xty = a_mtx.T @ b_vec

        return TypeContributionLstsq(xtx=batch_xtx, xty=batch_xty)

    @override
    def update(  # pylint: disable=arguments-differ
        self,
        type_counts: Int[Array, "batch_size n_types"] | Float[Array, "batch_size n_types"],
        values: Float[Array, "batch_size ..."],
        mask: Bool[Array, "batch_size"] | None = None,
        /,
    ) -> "TypeContributionLstsq":
        # Calculate stats for the incoming batch
        batch_metric = self.create(type_counts, values, mask)

        if self.is_empty:
            return batch_metric

        # Accumulate: Simple element-wise addition of the matrices
        return TypeContributionLstsq(
            xtx=self.xtx + batch_metric.xtx, xty=self.xty + batch_metric.xty
        )

    @override
    def merge(self, other: "TypeContributionLstsq") -> "TypeContributionLstsq":
        if self.is_empty:
            return other
        if other.is_empty:
            return self

        # Merging is just adding the sufficient statistics
        return TypeContributionLstsq(xtx=self.xtx + other.xtx, xty=self.xty + other.xty)

    @override
    def compute(self, regularization: float = 1e-6):
        """Computes the fit using the Moore-Penrose pseudo-inverse approach.

        This naturally handles rank-deficient cases (like fixed 50:50 ratios)
        by producing the minimum-norm solution, which assigns equal
        contributions to indistinguishable types.
        """
        if self.is_empty:
            raise RuntimeError("This metric is empty, cannot compute!")

        # Allow more mathsy names
        # pylint: disable=invalid-name

        np_ = utils.infer_backend(self.xtx)

        # Solve Normal Equation: (A.T A) x = A.T b
        # We solve for x in: xtx @ x = xty

        # 1. Since XtX is symmetric, eigh is more efficient and stable than SVD
        # s: eigenvalues, V: eigenvectors
        s, V = np_.linalg.eigh(self.xtx)

        # 2. Determine the threshold for 'zero' eigenvalues
        # Standard practice is a fraction of the largest eigenvalue
        max_s = np_.max(s)
        threshold = regularization * max_s

        # 3. Compute the pseudo-inverse of the eigenvalues
        # We only invert values above the threshold, others become 0.0
        s_inv = np_.where(s > threshold, 1.0 / s, 0.0)

        # 4. Reconstruct the solution: x = V @ diag(s_inv) @ V.T @ xty
        # This is the Moore-Penrose solution (minimum L2 norm)
        # Equivalent to: x = pinv(xtx) @ xty
        weights = V @ (s_inv[:, None] * (V.T @ self.xty))

        return weights


@jt.jaxtyped(typechecker=beartype.beartype)
class PropertyContributionLstsq(reax.Metric):
    """Metric to calculate the contribution of node or edge types to a global property
    using online linear regression.

    Args:
        type_key: Path to the type identifiers within the source (nodes or edges).
        property_key: Path to the global property to regress against.
        type_source: Whether the types originate from "nodes" or "edges".
        type_map: Optional mapping to transform raw types into continuous indices.
        normalize: If True, normalize counts and values by segment size.
        metric: Optional existing metric instance to continue from.

    Example:
        >>> metric = PropertyContributionLstsq(
        ...     type_key="atomic_numbers",
        ...     property_key="total_energy",
        ...     type_source="nodes",
        ...     normalize=True
        ... )
    """

    _type_key: typing.TreePath
    _property_key: typing.TreePath
    _type_source: str
    _type_map: Array | None
    _normalize: bool
    _metric: TypeContributionLstsq | None = None

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        type_key: typing.TreePathLike,
        property_key: typing.TreePathLike,
        type_source: Literal["nodes", "edges"] = "nodes",
        type_map: Sequence | Array | None = None,
        normalize: bool = False,
        *,
        metric: TypeContributionLstsq | None = None,
    ):
        self._type_key = _tree.path_from_str(type_key)
        self._property_key = _tree.path_from_str(property_key)
        self._type_source = type_source
        self._type_map = jnp.asarray(type_map) if type_map is not None else None
        self._normalize = normalize
        self._metric = metric

    @override
    def empty(self) -> "PropertyContributionLstsq":
        if self._metric is None:
            return self
        return type(self)(
            self._type_key,
            self._property_key,
            self._type_source,
            type_map=self._type_map,
            normalize=self._normalize,
        )

    @override
    def merge(self, other: "PropertyContributionLstsq") -> "PropertyContributionLstsq":
        if other._metric is None:  # pylint: disable=protected-access
            return self
        if self._metric is None:
            return other
        return PropertyContributionLstsq(
            self._type_key,
            self._property_key,
            self._type_source,
            type_map=self._type_map,
            normalize=self._normalize,
            metric=self._metric.merge(other._metric),  # pylint: disable=protected-access
        )

    @override
    def create(self, graphs: jraph.GraphsTuple, /, *_, **__) -> "PropertyContributionLstsq":
        val = self._fun(graphs)
        # Note: 'metric' is passed via TypeContributionLstsq.create
        return PropertyContributionLstsq(
            self._type_key,
            self._property_key,
            self._type_source,
            type_map=self._type_map,
            normalize=self._normalize,
            metric=TypeContributionLstsq.create(*val),
        )

    @override
    def update(self, graphs: jraph.GraphsTuple, /, *_, **__) -> "PropertyContributionLstsq":
        if self._metric is None:
            return self.create(graphs)
        val = self._fun(graphs)
        # Note: 'metric' is passed via self._metric.update
        return PropertyContributionLstsq(
            self._type_key,
            self._property_key,
            self._type_source,
            type_map=self._type_map,
            normalize=self._normalize,
            metric=self._metric.update(*val),
        )

    @override
    def compute(self, regularization: float = 1e-6):
        if self._metric is None:
            raise RuntimeError("Nothing to compute, metric is empty!")
        return self._metric.compute(regularization=regularization)

    def _fun(self, graphs: jraph.GraphsTuple, *_) -> tuple[
        Float[Array, "batch_size k"],
        Float[Array, "batch_size ..."],
        Bool[Array, "batch_size"] | None,
    ]:
        graph_dict = graphs._asdict()
        types = tree.get_by_path(graph_dict, (self._type_source,) + self._type_key)
        values = tree.get_by_path(graph_dict, ("globals",) + self._property_key)

        if self._type_map is None:
            num_classes = types.max().item() + 1
        else:
            types = nn_utils.vwhere(types, self._type_map)
            num_classes = len(self._type_map)

        one_hots = jax.nn.one_hot(types[:, 0], num_classes)
        segment_sizes = graphs.n_node if self._type_source == "nodes" else graphs.n_edge
        type_counts = graph_ops.segment_sum(one_hots, segment_sizes)

        if keys.MASK in graph_dict["globals"]:
            mask = graph_dict["globals"][keys.MASK]
        else:
            mask = None

        if self._normalize:
            # Normalize by segment sizes
            type_counts = jax.vmap(lambda numer, denom: numer / denom, (0, 0))(
                type_counts, segment_sizes
            )
            values = jax.vmap(lambda numer, denom: numer / denom, (0, 0))(values, segment_sizes)

        return type_counts, values, mask
