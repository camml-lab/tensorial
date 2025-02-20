import math
from typing import ClassVar, Literal, Optional, Sequence, TypeVar, Union

import beartype
import jax.numpy as jnp
import jax.typing
import jaxtyping as jt
import jraph
from pytray import tree
import reax

from . import _tree, _typing, keys

OutT = TypeVar("OutT")

__all__ = ("GraphMetric", "graph_metric")


@jt.jaxtyped(typechecker=beartype.beartype)
def graph_metric(
    metric: Union[str, reax.Metric, type[reax.Metric]],
    predictions: _typing.TreePathLike,
    targets: Optional[_typing.TreePathLike] = None,
    mask: Optional[Union[_typing.TreePathLike, Literal["auto"]]] = "auto",
    normalise_by: Optional[_typing.TreePathLike] = None,
) -> "GraphMetric":
    predictions_from = _tree.path_from_str(predictions)
    targets_from = _tree.path_to_str(targets) if targets is not None else predictions_from
    mask_from = _tree.path_to_str(mask) if mask is not None else None
    norm_by = _tree.path_to_str(normalise_by) if normalise_by is not None else None

    class _GraphMetric(GraphMetric):
        parent = reax.metrics.get(metric)
        pred_key = predictions_from
        target_key = targets_from
        mask_key = mask_from
        normalise_by = norm_by

    return _GraphMetric()


def mdiv(
    num: jax.typing.ArrayLike, denom: jax.typing.ArrayLike, where: jax.typing.ArrayLike = None
):
    """Divide that supports supplying a mask, where `False` values will just return the numerator"""
    # Use prod here because `IrrepsArray` doesn't have `.size`
    if math.prod(num.shape) != math.prod(denom.shape):
        raise ValueError(
            "Sizes of numerator and denominator must match, got {num.shape} and {denom.shape}"
        )
    if where is not None:
        where = reax.metrics.utils.prepare_mask(denom, where)
        denom = jnp.where(where, denom, 1.0)

    return num / denom.reshape(num.shape)


class GraphMetric(reax.Metric):
    parent: ClassVar[reax.Metric]
    pred_key: ClassVar[_typing.TreePathLike]
    target_key: ClassVar[Optional[_typing.TreePathLike]] = None
    mask_key: ClassVar[Optional[_typing.TreePathLike]] = "auto"
    normalise_by: ClassVar[Optional[_typing.TreePathLike]] = None

    _state: Optional[reax.Metric[OutT]]

    def __init__(self, state: Optional[reax.Metric[OutT]] = None):
        super().__init__()
        self._state = state

    @property
    def metric(self) -> Optional[reax.Metric[OutT]]:
        return self._state

    @property
    def is_empty(self) -> bool:
        return self._state is None

    def create(
        # pylint: disable=arguments-differ
        self,
        predictions: jraph.GraphsTuple,
        targets: Optional[jraph.GraphsTuple] = None,
    ) -> "GraphMetric":
        if targets is None:
            # In this case, the user is typically using a different key in the same graph
            targets = predictions

        pred = _tree.get(predictions, self.pred_key)
        targ = _tree.get(targets, self.target_key)

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

        if self.normalise_by is not None:
            pred = mdiv(pred, _tree.get(predictions, self.normalise_by), where=mask)
            targ = mdiv(targ, _tree.get(targets, self.normalise_by), where=mask)

        args = [pred, targ]
        if mask is not None:
            args.append(mask)

        return type(self)(self.parent.create(*args))

    def merge(self, other: "GraphMetric") -> "GraphMetric":
        if other.is_empty:
            return self
        if self.is_empty:
            return other

        return type(self)(self._state.merge(other._state))  # pylint: disable=protected-access

    def compute(self) -> OutT:
        if self.is_empty:
            raise RuntimeError("Cannot compute, metric is empty")

        return self._state.compute()


class AvgNumNeighboursByType(reax.Metric[dict[int, jax.Array]]):
    """
    Get the average number of node neighbours grouped by node type where the type is an integer
    found in G.nodes[type_field].
    """

    Averages = list[reax.metrics.Average]
    _type_field: str
    _node_types: jt.Int[jax.Array, "n_types"]
    _state: Optional[Averages]

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        node_types: Union[Sequence[int], jt.Int[jt.Array, "n_types"]],
        type_field: str = "type_id",
        state: Optional[Averages] = None,
    ):
        self._node_types = jnp.asarray(node_types)
        self._type_field = type_field
        self._state: Optional[AvgNumNeighboursByType.Averages] = state

    @property
    def is_empty(self) -> bool:
        return self._state is None

    def empty(self) -> "AvgNumNeighboursByType":
        if self.is_empty:
            return self

        return AvgNumNeighboursByType(self._node_types)

    def merge(self, other: "AvgNumNeighboursByType") -> "AvgNumNeighboursByType":
        if not jnp.all(self._node_types == other._node_types):  # pylint: disable=protected-access
            raise ValueError(
                f"Type maps must match, got {self._node_types} and {other._node_types}"  # pylint: disable=protected-access
            )

        if other.is_empty:  # pylint: disable=protected-access
            return self
        if self.is_empty:
            return other

        return AvgNumNeighboursByType(
            node_types=self._node_types,
            state=[
                avg.merge(other_avg)
                for avg, other_avg in zip(
                    self._state, other._state  # pylint: disable=protected-access
                )
            ],
        )

    def create(  # pylint: disable=arguments-differ
        self, graphs: jraph.GraphsTuple, *_
    ) -> "AvgNumNeighboursByType":
        state = self._calc_averages(graphs)  # pylint: disable=not-callable
        return AvgNumNeighboursByType(
            node_types=self._node_types, type_field=self._type_field, state=state
        )

    def update(  # pylint: disable=arguments-differ
        self, graphs: jraph.GraphsTuple, *_
    ) -> "AvgNumNeighboursByType":
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

        types = tree.get_by_path(graph_dict, ("nodes", self._type_field))
        # Transform the type numbers from whatever they are to 0, 1, 2....
        vwhere = jax.vmap(lambda num: jnp.argwhere(num == self._node_types, size=1)[0])
        types = vwhere(types)[:, 0]

        counts = jnp.bincount(graphs.senders, length=jnp.sum(graphs.n_node).item())
        mask = reax.metrics.utils.prepare_mask(counts, graphs.nodes.get(keys.MASK))
        mask = mask if mask is not None else True

        num_classes = len(self._node_types)
        return [
            reax.metrics.Average.create(counts, mask & (types == idx)) for idx in range(num_classes)
        ]
