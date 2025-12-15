from collections.abc import Mapping, Sequence

import beartype
import jax
import jax.numpy as jnp
import jaxtyping as jt
import jraph
from pytray import tree
import reax
from typing_extensions import override

from tensorial.typing import Array

from . import keys
from .. import graph_ops
from .. import keys as graph_keys
from .. import metrics
from ... import nn_utils, utils

__all__ = (
    "AllAtomicNumbers",
    "NumSpecies",
    "ForceStd",
    "AvgNumNeighbours",
    "AvgNumNeighboursByAtomType",
    "TypeContributionLstsq",
    "EnergyContributionLstsq",
    "EnergyPerAtomLstsq",
)


def get(mapping: Mapping, key: str):
    try:
        return mapping[key]
    except KeyError:
        raise reax.exceptions.DataNotFound(f"Missing key: {key}") from None


AllAtomicNumbers = reax.metrics.Unique.from_fun(
    lambda graph, *_: (get(graph.nodes, keys.ATOMIC_NUMBERS), graph.nodes.get(graph_keys.MASK)),
    name="AtomicNumbers",
)


NumSpecies = reax.metrics.NumUnique.from_fun(
    lambda graph: (get(graph.nodes, keys.ATOMIC_NUMBERS), graph.nodes.get(graph_keys.MASK)),
    name="Species",
)


ForceStd = reax.metrics.Std.from_fun(
    lambda graph: (get(graph.nodes, keys.FORCES), graph.nodes.get(graph_keys.MASK)), name="Force"
)


AvgNumNeighbours = reax.metrics.Average.from_fun(
    lambda graph, *_: (
        jnp.bincount(graph.senders, length=jnp.sum(graph.n_node)),
        graph.nodes.get(graph_keys.MASK),
    )
)


class EnergyPerAtomLstsq(reax.metrics.FromFun):
    """Calculate the least squares estimate of the energy per atom"""

    metric = reax.metrics.LeastSquaresEstimate

    @staticmethod
    def func(graph, *_):
        return graph.n_node.reshape(-1, 1), graph.globals[keys.TOTAL_ENERGY].reshape(-1)

    def compute(self) -> jax.Array:
        return super().compute().reshape(())


class TypeContributionLstsq(reax.metrics.Metric[Array]):
    """Online Least Squares Metric.

    Uses 'Sufficient Statistics' (XtX, Xty) to perform linear regression
    without storing the entire dataset history.
    """

    # XtX: The Gram Matrix (A.T @ A) -> Shape: (n_types, n_types)
    xtx: jt.Float[jax.Array, "n_types n_types"] | None = None

    # Xty: The Moment Vector (A.T @ b) -> Shape: (n_types, ...)
    xty: jt.Float[jax.Array, "n_types ..."] | None = None

    @property
    def is_empty(self):
        return self.xtx is None

    @classmethod
    @override
    def empty(cls) -> "TypeContributionLstsq":  # pylint: disable=arguments-differ
        return cls()

    @classmethod
    @jt.jaxtyped(typechecker=beartype.beartype)
    @override
    def create(  # pylint: disable=arguments-differ
        cls,
        type_counts: jt.Int[Array, "batch_size n_types"] | jt.Float[Array, "batch_size n_types"],
        values: jt.Float[Array, "batch_size ..."],
        mask: jt.Bool[Array, "batch_size"] | None = None,
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

        return cls(xtx=batch_xtx, xty=batch_xty)

    @jt.jaxtyped(typechecker=beartype.beartype)
    @override
    def update(  # pylint: disable=arguments-differ
        self,
        type_counts: jt.Int[Array, "batch_size n_types"] | jt.Float[Array, "batch_size n_types"],
        values: jt.Float[Array, "batch_size ..."],
        mask: jt.Bool[Array, "batch_size"] | None = None,
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
        if self.is_empty:
            raise RuntimeError("This metric is empty, cannot compute!")

        np_ = utils.infer_backend(self.xtx)

        # Solve Normal Equation: (A.T A) x = A.T b
        # We solve for x in: xtx @ x = xty

        # Add small ridge regularization for numerical stability
        # (prevents crash if matrix is singular or data was empty)
        eye = np_.eye(self.xtx.shape[0])
        safe_xtx = self.xtx + (eye * regularization)

        return np_.linalg.solve(safe_xtx, self.xty)


class EnergyContributionLstsq(reax.Metric):
    _type_map: jt.Array
    _metric: TypeContributionLstsq | None = None

    def __init__(self, type_map: Sequence | Array, metric: TypeContributionLstsq = None):
        if type_map is None:
            raise ValueError("Must supply a value type_map")
        self._type_map = jnp.asarray(type_map)
        self._metric = metric

    @override
    def empty(self) -> "EnergyContributionLstsq":
        if self._metric is None:
            return self

        return EnergyContributionLstsq(self._type_map)

    @override
    def merge(self, other: "EnergyContributionLstsq") -> "EnergyContributionLstsq":
        if other._metric is None:  # pylint: disable=protected-access
            return self
        if self._metric is None:
            return other

        return type(self)(
            type_map=self._type_map,
            metric=self._metric.merge(other._metric),  # pylint: disable=protected-access
        )

    @override
    def create(  # pylint: disable=arguments-differ
        self, graphs: jraph.GraphsTuple, *_
    ) -> "EnergyContributionLstsq":
        val = self._fun(graphs)  # pylint: disable=not-callable
        return type(self)(type_map=self._type_map, metric=TypeContributionLstsq.create(*val))

    @override
    def update(  # pylint: disable=arguments-differ
        self, graphs: jraph.GraphsTuple, *_
    ) -> "EnergyContributionLstsq":
        if self._metric is None:
            return self.create(graphs)

        val = self._fun(graphs)  # pylint: disable=not-callable
        return EnergyContributionLstsq(type_map=self._type_map, metric=self._metric.update(*val))

    @override
    def compute(self):
        if self._metric is None:
            raise RuntimeError("Nothing to compute, metric is empty!")

        return self._metric.compute()

    @jt.jaxtyped(typechecker=beartype.beartype)
    def _fun(self, graphs: jraph.GraphsTuple, *_) -> tuple[
        jt.Float[Array, "batch_size k"],
        jt.Float[Array, "batch_size 1"],
        jt.Bool[Array, "batch_size"] | None,
    ]:
        graph_dict = graphs._asdict()
        num_nodes = graphs.n_node

        try:
            types = tree.get_by_path(graph_dict, ("nodes", keys.ATOMIC_NUMBERS))
        except KeyError:
            raise reax.exceptions.DataNotFound(
                f"Missing key: {('nodes', keys.TOTAL_ENERGY)}"
            ) from None

        if self._type_map is None:
            num_classes = types.max().item() + 1  # Assume the types go 0,1,2...N
        else:
            # Transform the atomic numbers from whatever they are to 0, 1, 2....
            types = nn_utils.vwhere(types, self._type_map)
            num_classes = len(self._type_map)

        one_hots = jax.nn.one_hot(types, num_classes)

        # TODO: make it so we don't need to set the value in the graph
        one_hot_field = ("type_one_hot",)
        tree.set_by_path(graphs.nodes, one_hot_field, one_hots)
        type_counts = graph_ops.graph_segment_reduce(
            graphs, ("nodes",) + one_hot_field, reduction="sum"
        )

        # Predicting values
        try:
            values = tree.get_by_path(graph_dict, ("globals", keys.TOTAL_ENERGY))
        except KeyError:
            raise reax.exceptions.DataNotFound(
                f"Missing key: {('globals', keys.TOTAL_ENERGY)}"
            ) from None

        if graph_keys.MASK in graph_dict["globals"]:
            mask = graph_dict["globals"][graph_keys.MASK]
        else:
            mask = None

        # Normalise by number of nodes
        type_counts = jax.vmap(lambda numer, denom: numer / denom, (0, 0))(type_counts, num_nodes)
        values = jax.vmap(lambda numer, denom: numer / denom, (0, 0))(values, num_nodes)

        return type_counts, values, mask


class AvgNumNeighboursByAtomType(metrics.AvgNumNeighboursByType):
    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        atom_types: Sequence[int] | jt.Int[Array, "n_types"],
        type_field: str = keys.ATOMIC_NUMBERS,
        state: metrics.AvgNumNeighboursByType.Averages | None = None,
    ):
        super().__init__(atom_types, type_field, state)
