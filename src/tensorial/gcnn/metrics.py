import math
from typing import ClassVar, Optional, TypeVar, Union

import jax.numpy as jnp
import jax.typing
import jraph
import reax

from tensorial import nn_utils

from . import _tree, _typing

OutT = TypeVar("OutT")

__all__ = ("GraphMetric", "graph_metric")


def graph_metric(
    metric: Union[str, reax.Metric, type[reax.Metric]],
    predictions: _typing.TreePathLike,
    targets: Optional[_typing.TreePathLike] = None,
    mask: Optional[_typing.TreePathLike] = None,
    normalise_by: Optional[_typing.TreePathLike] = None,
) -> type["GraphMetric"]:

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

    return _GraphMetric


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
        where = nn_utils.prepare_mask(where, denom)
        denom = jnp.where(where, denom, 1.0)
    return num / denom.reshape(num.shape)


class GraphMetric(reax.Metric):

    parent: ClassVar[reax.Metric]
    pred_key: ClassVar[_typing.TreePathLike]
    target_key: ClassVar[Optional[_typing.TreePathLike]] = None
    mask_key: ClassVar[Optional[_typing.TreePathLike]] = None
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
            # todo: add check to this for what happens if mask doesn't exist
            mask = _tree.get(predictions, self.mask_key)

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
