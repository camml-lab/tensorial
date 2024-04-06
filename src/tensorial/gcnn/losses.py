# -*- coding: utf-8 -*-
import abc
from typing import Callable, Dict, Sequence, Tuple, Union

import equinox
import jax
import jax.numpy as jnp
import jraph
import optax.losses
from pytray import tree

from . import utils

__all__ = ('SimpleLossFn', 'GraphLoss', 'WeightedLoss')

SimpleLossFn = Callable[[jax.Array, jax.Array], jax.Array]


class GraphLoss(equinox.Module):
    _label: str

    def __init__(self, label: str):
        self._label = label

    def label(self) -> str:
        """Get a label for this loss function"""
        return self._label

    def __call__(self, predictions: jraph.GraphsTuple, targets: jraph.GraphsTuple = None) -> jax.Array:
        """Return the scalar loss between predictions and targets"""
        if targets is None:
            targets = predictions
        return self._call(predictions, targets)

    @abc.abstractmethod
    def _call(self, predictions: jraph.GraphsTuple, targets: jraph.GraphsTuple) -> jax.Array:
        """Return the scalar loss between predictions and targets"""


class Loss(GraphLoss):
    """Simple loss function that passes values from the graph to a function taking numerical values such as optax
    losses"""
    _loss_fn: SimpleLossFn
    _prediction_field: utils.TreePath
    _target_field: utils.TreePath
    _reduction: str

    def __init__(
        self,
        prediction_field: str,
        target_field: str = None,
        loss_fn: Union[str, SimpleLossFn] = optax.squared_error,
        reduction: str = 'mean',
        label: str = None
    ):
        if reduction not in ('sum', 'mean', None):
            raise ValueError(f"Invalid reduction, must be one of 'sum', 'mean', `None`, got {reduction}")

        self._loss_fn = _get_loss_fn(loss_fn)
        self._prediction_field = utils.path_from_str(prediction_field)
        self._target_field = utils.path_from_str(target_field or prediction_field)
        self._reduction = reduction
        super().__init__(label or utils.path_to_str(self._prediction_field))

    def _call(self, predictions: jraph.GraphsTuple, targets: jraph.GraphsTuple) -> jax.Array:
        loss = self._loss_fn(
            tree.get_by_path(predictions._asdict(), self._prediction_field),
            tree.get_by_path(targets._asdict(), self._target_field),
        )
        if self._reduction == 'mean':
            loss = loss.mean()
        elif self._reduction == 'sum':
            loss = loss.sum()

        return loss


class WeightedLoss(GraphLoss):
    _weights: jax.Array
    _loss_fns: Sequence[GraphLoss]

    def __init__(
        self,
        losses: Sequence,
    ):
        super().__init__('weighted loss')
        weights = []
        loss_fns = []
        for loss in losses:
            weight, loss_fn = _loss_and_weight(loss)
            weights.append(weight)
            loss_fns.append(loss_fn)

        self._weights = jnp.array(weights)
        self._loss_fns = loss_fns

    def _call(self, predictions: jraph.GraphsTuple, targets: jraph.GraphsTuple) -> float:
        # Calculate the loss for each function
        losses = jnp.array(list(map(lambda loss_fn: loss_fn(predictions, targets), self._loss_fns)))
        return jnp.dot(self._weights, losses)

    def loss_with_contributions(self, predictions: jraph.GraphsTuple,
                                target: jraph.GraphsTuple) -> Tuple[float, Dict[str, float]]:
        # Calculate the loss for each function
        losses = jax.array(list(map(lambda loss_fn: loss_fn(predictions, target), self._loss_fns)))
        # Group the contributions into a dictionary keyed by the label
        contribs = dict(zip(list(map(GraphLoss.label, self._loss_fns)), losses))

        return jnp.dot(self._weights, losses), contribs


def _loss_and_weight(entry: Union[Tuple[float, GraphLoss], dict]) -> Tuple[float, GraphLoss]:
    if isinstance(entry, tuple):
        return entry
    if isinstance(entry, dict):
        return entry['weight'], entry['fn']

    raise ValueError(f'Unknown loss and weight type: {type(entry).__name__}')


def _get_loss_fn(loss_fn: SimpleLossFn):
    if isinstance(loss_fn, str):
        return getattr(optax.losses, loss_fn)

    return loss_fn
