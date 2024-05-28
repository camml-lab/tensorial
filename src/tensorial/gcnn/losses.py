# -*- coding: utf-8 -*-
import abc
from typing import Callable, Optional, Sequence, Tuple, Union

import equinox
import jax
import jax.numpy as jnp
import jraph
import optax.losses
from pytray import tree

import tensorial
from tensorial import nn_utils

from . import keys, utils

__all__ = ("PureLossFn", "GraphLoss", "WeightedLoss", "Loss")

# A pure loss function that doesn't know about graphs, just takes arrays and produces a loss array
PureLossFn = Callable[[jax.Array, jax.Array], jax.Array]


class GraphLoss(equinox.Module):
    _label: str

    def __init__(self, label: str):
        self._label = label

    def label(self) -> str:
        """Get a label for this loss function"""
        return self._label

    def __call__(
        self, predictions: jraph.GraphsTuple, targets: jraph.GraphsTuple = None
    ) -> jax.Array:
        """Return the scalar loss between predictions and targets"""
        if targets is None:
            targets = predictions
        return self._call(predictions, targets)

    @abc.abstractmethod
    def _call(self, predictions: jraph.GraphsTuple, targets: jraph.GraphsTuple) -> jax.Array:
        """Return the scalar loss between predictions and targets"""


class Loss(GraphLoss):
    """
    Simple loss function that passes values from the graph to a function taking numerical values
    such as optax losses
    """

    _loss_fn: PureLossFn
    _prediction_field: utils.TreePath
    _target_field: utils.TreePath
    _mask_field: Optional[utils.TreePath]
    _reduction: str

    def __init__(
        self,
        field: str,
        target_field: str = None,
        loss_fn: Union[str, PureLossFn] = optax.squared_error,
        reduction: str = "mean",
        label: str = None,
        mask_field: str = None,
    ):
        if reduction not in ("sum", "mean", None):
            raise ValueError(
                f"Invalid reduction, must be one of 'sum', 'mean', `None`, got {reduction}"
            )

        self._loss_fn = _get_pure_loss_fn(loss_fn)
        self._prediction_field = utils.path_from_str(field)
        self._target_field = utils.path_from_str(target_field or field)
        if mask_field is not None:
            self._mask_field = utils.path_from_str(mask_field)
        else:
            self._mask_field = None
        self._reduction = reduction
        super().__init__(label or utils.path_to_str(self._prediction_field))

    def _call(self, predictions: jraph.GraphsTuple, targets: jraph.GraphsTuple) -> jax.Array:
        predictions_dict = predictions._asdict()
        mask = predictions_dict[self._prediction_field[0]].get(keys.MASK)

        _predictions = tensorial.as_array(
            tree.get_by_path(predictions_dict, self._prediction_field)
        )
        _targets = tensorial.as_array(tree.get_by_path(targets._asdict(), self._target_field))

        loss = self._loss_fn(_predictions, _targets)
        num_elements = loss.size

        if mask is not None:
            mask = nn_utils.prepare_mask(mask, loss)
        # Check for the presence of a user-defined mask
        if self._mask_field:
            user_mask = nn_utils.prepare_mask(
                tensorial.as_array(tree.get_by_path(targets._asdict(), self._mask_field)),
                loss,
            )
            if mask is None:
                mask = user_mask
            else:
                mask = mask & user_mask

        if mask is not None:
            loss = jnp.where(mask, loss, 0.0)  # Zero out the masked elements
            # Now calculate the number of elements that were masked so that we get the correct mean
            num_elements = jnp.array([mask.sum(), *loss.shape[1:]]).prod()

        if self._reduction == "mean":
            loss = loss.sum() / num_elements
        elif self._reduction == "sum":
            loss = loss.sum()

        return loss


class WeightedLoss(GraphLoss):
    _weights: Tuple[float, ...]
    _loss_fns: Sequence[GraphLoss]

    def __init__(
        self,
        weights: Sequence[float],
        loss_fns: Sequence[GraphLoss],
    ):
        super().__init__("weighted loss")
        for loss in loss_fns:
            if not isinstance(loss, GraphLoss):
                raise ValueError(
                    f"loss_fns must all be subclasses of GraphLoss, got {type(loss).__name__}"
                )

        if len(weights) != len(loss_fns):
            raise ValueError(
                f"the number of weights and loss functions must be equal, got {len(weights)} and "
                f"{len(loss_fns)}"
            )

        self._weights = tuple(
            weights
        )  # We have to use a list here, otherwise jax will treat this as a dynamic type
        self._loss_fns = loss_fns

    @property
    def weights(self):
        return jax.lax.stop_gradient(jnp.array(self._weights))

    def _call(self, predictions: jraph.GraphsTuple, targets: jraph.GraphsTuple) -> float:
        # Calculate the loss for each function
        losses = jnp.array(list(map(lambda loss_fn: loss_fn(predictions, targets), self._loss_fns)))
        return jnp.dot(self.weights, losses)

    def loss_with_contributions(
        self, predictions: jraph.GraphsTuple, target: jraph.GraphsTuple
    ) -> Tuple[float, dict[str, float]]:
        # Calculate the loss for each function
        losses = jax.array(list(map(lambda loss_fn: loss_fn(predictions, target), self._loss_fns)))
        # Group the contributions into a dictionary keyed by the label
        contribs = dict(zip(list(map(GraphLoss.label, self._loss_fns)), losses))

        return jnp.dot(self.weights, losses), contribs


def _get_pure_loss_fn(loss_fn: Union[str, PureLossFn]) -> PureLossFn:
    if isinstance(loss_fn, str):
        return getattr(optax.losses, loss_fn)
    if isinstance(loss_fn, Callable):
        return loss_fn

    raise ValueError(f"Unknown loss function type: {type(loss_fn).__name__}")
