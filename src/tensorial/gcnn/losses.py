import abc
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Literal, Optional, Union

import beartype
import equinox
import jax
import jax.numpy as jnp
import jaxtyping as jt
import jraph
import optax.losses
from pytray import tree
import reax

from . import keys, utils
from .. import base

if TYPE_CHECKING:
    from tensorial import gcnn

__all__ = "PureLossFn", "GraphLoss", "WeightedLoss", "Loss"

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
    _prediction_field: "gcnn.typing.TreePath"
    _target_field: "gcnn.typing.TreePath"
    _mask_field: "Optional[gcnn.typing.TreePath]"
    _reduction: Optional[Literal["sum", "mean"]]

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        field: str,
        target_field: str = None,
        loss_fn: Union[str, PureLossFn] = optax.squared_error,
        *,
        reduction: Optional[Literal["sum", "mean"]] = "mean",
        label: str = None,
        mask_field: Optional[str] = None,
    ):
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

        _predictions = base.as_array(tree.get_by_path(predictions_dict, self._prediction_field))
        _targets = base.as_array(tree.get_by_path(targets._asdict(), self._target_field))

        loss = self._loss_fn(_predictions, _targets)

        # If there is a mask in the graph, then use it by default
        mask = predictions_dict[self._prediction_field[0]].get(keys.MASK)
        mask = reax.metrics.utils.prepare_mask(loss, mask)

        # Now, check for the presence of a user-defined mask
        if self._mask_field:
            user_mask = base.as_array(tree.get_by_path(targets._asdict(), self._mask_field))
            user_mask = reax.metrics.utils.prepare_mask(loss, user_mask)
            if mask is None:
                mask = user_mask
            else:
                mask = mask & user_mask

        # Now calculate the number of elements that were masked so that we get the correct mean
        num_elements = (
            loss.size if mask is None else jnp.array([mask.sum(), *loss.shape[1:]]).prod()
        )

        if self._reduction == "mean":
            loss = loss / num_elements
            # loss = loss.sum(where=mask) / num_elements
        # elif self._reduction == "sum":
        loss = loss.sum(where=mask)

        return loss


class WeightedLoss(GraphLoss):
    _weights: tuple[float, ...]
    _loss_fns: Sequence[GraphLoss]

    def __init__(
        self,
        loss_fns: Sequence[GraphLoss],
        weights: Optional[Sequence[float]] = None,
    ):
        super().__init__("weighted loss")
        for loss in loss_fns:
            if not isinstance(loss, GraphLoss):
                raise ValueError(
                    f"loss_fns must all be subclasses of GraphLoss, got {type(loss).__name__}"
                )

        if weights is None:
            weights = (1.0,) * len(loss_fns)
        else:
            if len(weights) != len(loss_fns):
                raise ValueError(
                    f"the number of weights and loss functions must be equal, got {len(weights)} "
                    f"and {len(loss_fns)}"
                )

        self._weights = tuple(
            weights
        )  # We have to use a tuple here, otherwise jax will treat this as a dynamic type
        self._loss_fns = loss_fns

    @property
    def weights(self):
        return jax.lax.stop_gradient(jnp.array(self._weights))

    def _call(self, predictions: jraph.GraphsTuple, targets: jraph.GraphsTuple) -> jax.Array:
        # Calculate the loss for each function
        losses = jnp.array(list(map(lambda loss_fn: loss_fn(predictions, targets), self._loss_fns)))
        return jnp.dot(self.weights, losses)

    def loss_with_contributions(
        self, predictions: jraph.GraphsTuple, target: jraph.GraphsTuple
    ) -> tuple[jax.Array, dict[str, float]]:
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
