from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar, TypeVar, cast

import beartype
import jax
import jax.numpy as jnp
import jaxtyping as jt
from jaxtyping import Array, Int
import jraph
from pytray import tree
import reax
from reax.utils import masking
from typing_extensions import override

from ... import nn_utils
from .. import _tree, keys
from ..graph_ops import mdiv  # noqa: F401  (re-exported for backwards compatibility)

if TYPE_CHECKING:
    from typing import Literal

    from tensorial import gcnn

OutT = TypeVar("OutT")

__all__ = "GraphMetric", "graph_metric", "AvgNumNeighboursByType"


@jt.jaxtyped(typechecker=beartype.beartype)
def graph_metric(
    metric: str | reax.Metric | type[reax.Metric],
    predictions: "gcnn.typing.TreePathLike",
    targets: "gcnn.TreePathLike | None" = None,
    mask: "gcnn.TreePathLike | Literal['auto'] | None" = "auto",
    normalise_by: "gcnn.TreePathLike | None" = None,
    optional: bool = False,
) -> "GraphMetric":
    """Create a metric that is evaluated on fields of a graph.

    Padding masks, and the per-field mask next to the target (see
    :func:`~tensorial.gcnn.keys.mask_for`), are applied when present.

    Args:
        metric: the metric to evaluate (or its name)
        predictions: path to the predicted values
        targets: path to the target values, if the metric takes them
        mask: path to the mask, or ``"auto"`` for the padding mask of the predictions
        normalise_by: path to values to divide predictions and targets by
        optional: if ``True``, the metric is skipped for data that lacks the predicted or
            target field (e.g. stresses of molecules) instead of raising an error
    """
    predictions_from = _tree.path_from_str(predictions)
    targets_from = _tree.path_to_str(targets) if targets is not None else None
    mask_from = _tree.path_to_str(mask) if mask is not None else None
    norm_by = _tree.path_to_str(normalise_by) if normalise_by is not None else None

    class _GraphMetric(GraphMetric):
        parent = reax.metrics.get(metric)
        pred_key = predictions_from
        target_key = targets_from
        mask_key = mask_from
        normalise_by = norm_by
        is_optional = optional

    return _GraphMetric()


class GraphMetric(reax.Metric):
    parent: ClassVar[reax.Metric]
    pred_key: "ClassVar[gcnn.typing.TreePathLike]"
    target_key: "ClassVar[gcnn.typing.TreePathLike | None]" = None
    mask_key: "ClassVar[gcnn.typing.TreePathLike | None]" = "auto"
    normalise_by: "ClassVar[gcnn.typing.TreePathLike | None]" = None
    is_optional: ClassVar[bool] = False

    _state: reax.Metric[OutT] | None

    def __init__(self, state: reax.Metric[OutT] | None = None):
        super().__init__()
        self._state = state

    @property
    def metric(self) -> reax.Metric[OutT] | None:
        return self._state

    @property
    def is_empty(self) -> bool:
        return self._state is None

    @override
    def create(
        self,
        predictions: jraph.GraphsTuple,
        targets: jraph.GraphsTuple | None = None,
    ) -> "GraphMetric":
        if targets is None:
            # In this case, the user is typically using a different key in the same graph
            targets = predictions

        if self.is_optional and not (
            _tree.has_path(predictions, self.pred_key)
            and (self.target_key is None or _tree.has_path(targets, self.target_key))
        ):
            # Lets a ``MetricCollection`` created with ``ignore_missing=True`` leave us out
            raise reax.exceptions.DataNotFound(
                f"No '{self.pred_key}' or '{self.target_key}' in the data for an optional metric"
            )

        pred = _tree.get(predictions, self.pred_key)

        mask = None
        if self.mask_key is not None:
            if self.mask_key == "auto":
                pred_key = _tree.path_from_str(self.pred_key)
                mask_key = pred_key[:-1] + ("mask",)
            else:
                mask_key = self.mask_key

            try:
                mask = _tree.get(predictions, mask_key)
            except KeyError:
                mask = None

        if self.target_key is not None:
            # Leave out entries that only hold a filled-in placeholder (e.g. zero stress)
            field_mask = _tree.get_field_mask(targets, self.target_key)
            if field_mask is not None:
                mask = field_mask if mask is None else mask & field_mask

        if self.normalise_by is not None:
            pred = mdiv(pred, _tree.get(predictions, self.normalise_by), where=mask)

        args = [pred]
        # If there is a target field, add that to the argument list
        if self.target_key:
            targ = _tree.get(targets, self.target_key)
            if self.normalise_by is not None:
                targ = mdiv(targ, _tree.get(targets, self.normalise_by), where=mask)

            args.append(targ)

        kwargs = {} if mask is None else {"mask": mask}

        return type(self)(self.parent.create(*args, **kwargs))

    @override
    def merge(self, other: "GraphMetric") -> "GraphMetric":
        if other.is_empty:
            return self
        if self.is_empty:
            return other

        return type(self)(self._state.merge(other._state))  # noqa: SLF001

    @override
    def compute(self) -> OutT:
        if self.is_empty:
            raise RuntimeError("Cannot compute, metric is empty")

        return self._state.compute()

    @override
    def reduce(self, axis: int = 0) -> "reax.Metric[OutT]":
        if self.is_empty:
            raise RuntimeError("Cannot compute, metric is empty")

        return type(self)(self._state.reduce(axis=axis))


class AvgNumNeighboursByType(reax.Metric[dict[int, jax.Array]]):
    """Get the average number of node neighbours grouped by node type where the type is an integer
    found in G.nodes[type_field].
    """

    Averages = list[reax.metrics.Average]
    _type_field: str
    _node_types: Int[Array, "n_types"]
    _state: Averages | None

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        node_types: Sequence[int] | Int[Array, "n_types"],
        type_field: str = "type_id",
        state: Averages | None = None,
    ):
        """
        Initializes the AvgNumNeighboursByType class with node types, type field, and optional
        state.

        Args:
            node_types: Sequence of integers representing the types of nodes or an array with shape
                (n_types,).
            type_field: String indicating the field name used to identify node types, defaults to
                "type_id".
            state: Optional Averages object containing precomputed state information.

        Raises:
            ValueError: If node_types is empty or contains invalid values.
            TypeError: If type_field is not a string or state is not of type Averages or None.
        """
        self._node_types = jnp.asarray(node_types)
        self._type_field = type_field
        self._state: AvgNumNeighboursByType.Averages | None = state

    @property
    def is_empty(self) -> bool:
        return self._state is None

    def empty(self) -> "AvgNumNeighboursByType":
        if self.is_empty:
            return self

        return AvgNumNeighboursByType(self._node_types)

    def merge(self, other: "AvgNumNeighboursByType") -> "AvgNumNeighboursByType":
        if self._node_types.shape != other._node_types.shape or not bool(  # noqa: SLF001
            jnp.all(self._node_types == other._node_types)  # noqa: SLF001
        ):
            raise ValueError(
                f"Type maps must match, got {self._node_types} and {other._node_types}"  # noqa: SLF001
            )

        if other.is_empty:
            return self
        if self.is_empty:
            return other

        return AvgNumNeighboursByType(
            node_types=self._node_types,
            state=[
                avg.merge(other_avg)
                for avg, other_avg in zip(
                    self._state,
                    other._state,  # noqa: SLF001
                )
            ],
        )

    def create(self, graphs: jraph.GraphsTuple, *_) -> "AvgNumNeighboursByType":
        state = self._calc_averages(graphs)
        return AvgNumNeighboursByType(
            node_types=self._node_types, type_field=self._type_field, state=state
        )

    def update(self, graphs: jraph.GraphsTuple, *_) -> "AvgNumNeighboursByType":
        if self.is_empty:
            return self.create(graphs)

        # Create the updated state
        state = [
            avg.merge(other_avg) for avg, other_avg in zip(self._state, self._calc_averages(graphs))
        ]
        return AvgNumNeighboursByType(node_types=self._node_types, state=state)

    def compute(self) -> dict[int, jax.Array]:
        if self.is_empty:
            raise RuntimeError("Nothing to compute, metric is empty!")

        return {
            type_id: avg.compute() for type_id, avg in zip(self._node_types.tolist(), self._state)
        }

    @jt.jaxtyped(typechecker=beartype.beartype)
    def _calc_averages(self, graphs: jraph.GraphsTuple, *_) -> Averages:
        graph_dict = graphs._asdict()

        types = tree.get_by_path(graph_dict, ("nodes", self._type_field))[:, 0]
        # Transform the type numbers from whatever they are to 0, 1, 2....
        types = nn_utils.vwhere(types, self._node_types)

        counts = jnp.bincount(graphs.senders, length=jnp.sum(graphs.n_node).item())
        nodes = cast("dict[str, Any]", graphs.nodes)
        node_mask = nodes.get(keys.MASK)
        mask = True if node_mask is None else masking.prepare_mask(counts, node_mask)

        num_classes = len(self._node_types)
        return [
            reax.metrics.Average.create(counts, mask & (types == idx)) for idx in range(num_classes)
        ]
