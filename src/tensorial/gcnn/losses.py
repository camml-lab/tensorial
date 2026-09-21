"""Graph-aware loss functions for `jraph.GraphsTuple` model outputs.

Wraps pure (array-level) loss functions with the ability to reduce over batches of
variable-size graphs, optionally apply per-graph weights, and expose per-graph losses
for logging/monitoring.
"""

import abc
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Final, Literal, Optional

import beartype
import equinox
import jax
import jax.numpy as jnp
import jaxtyping as jt
from jaxtyping import Array, Bool, Float, Int
import jraph
import optax.losses
from pytray import tree
import reax

from . import _tree, graph_ops, keys, typing, utils
from .. import base

if TYPE_CHECKING:
    from tensorial import gcnn

__all__ = "PureLossFn", "GraphLoss", "WeightedLoss", "Loss"

# A pure loss function that doesn't know about graphs, just takes arrays and produces a loss array
PureLossFn = Callable[[jax.Array, jax.Array], jax.Array]


class GraphLoss(equinox.Module):
    """Base class for losses over batches of `jraph.GraphsTuple`.

    Subclasses implement `_call` (and optionally `loss_with_contributions`) to
    combine a batch of predictions with a batch of targets into a scalar loss.
    """

    _label: str

    def __init__(self, label: str):
        self._label = label

    def label(self) -> str:
        """Get a label for this loss function"""
        return self._label

    @property
    def reduction(self) -> Literal["sum", "mean"]:
        """How the value returned by this loss has been reduced over the batch.

        Needed by anything that aggregates losses over several batches (e.g. logging an epoch
        average), as a mean and a total have to be combined differently.
        """
        return "mean"

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
    """Simple loss function that passes values from the graph to a function taking numerical values
    such as optax losses
    """

    _loss_fn: PureLossFn
    _target_field: "gcnn.typing.TreePath"
    _prediction_field: "gcnn.typing.TreePath"
    _mask_field: "Optional[gcnn.typing.TreePath]"
    _reduction: Literal["sum", "mean"]

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        loss_fn: str | PureLossFn,
        targets: str,
        predictions: str | None = None,
        *,
        reduction: Literal["sum", "mean"] = "mean",
        label: str = None,
        mask_field: str | None = None,
    ):
        """
        Initializes the loss function wrapper with specified parameters.

        This constructor sets up a loss function wrapper that can be used to compute
        loss values based on target and prediction fields. It supports various loss
        functions and provides options for reduction and masking.

        Args:
            loss_fn: The loss function to use, either as a string identifier or a
                callable implementing the PureLossFn protocol.
            targets: The path to the target field in the data structure.
            predictions: The path to the prediction field in the data structure. If
                None, the target field path will be used.
            reduction: The reduction method to apply to the computed losses. Either
                "sum" or "mean".
            label: The label to use for the loss function. If None, the
                prediction field path will be used as the label.
            mask_field: The path to the mask field in the data structure. If None,
                no masking will be applied.

        Raises:
            ValueError: If the reduction method is not "sum" or "mean".
            TypeError: If the loss function is not a valid string or callable.
        """
        self._loss_fn = _get_pure_loss_fn(loss_fn)
        self._target_field: Final[typing.TreePath] = utils.path_from_str(targets)
        self._prediction_field: Final[typing.TreePath] = utils.path_from_str(predictions or targets)
        if mask_field is not None:
            self._mask_field = utils.path_from_str(mask_field)
        else:
            self._mask_field = None
        self._reduction = reduction
        super().__init__(label or utils.path_to_str(self._prediction_field))

    @property
    def reduction(self) -> Literal["sum", "mean"]:
        """How per-graph losses are combined into the batch scalar."""
        return self._reduction

    def _call(self, predictions: jraph.GraphsTuple, targets: jraph.GraphsTuple) -> jax.Array:
        """Compute the scalar loss by applying the wrapped loss function to the fields."""
        predictions_dict = predictions._asdict()

        pred_values = base.as_array(tree.get_by_path(predictions_dict, self._prediction_field))
        target_values = base.as_array(tree.get_by_path(targets._asdict(), self._target_field))

        loss = self._loss_fn(pred_values, target_values)

        # If there is a mask in the graph, then use it by default
        mask = _tree.get_mask(targets, self._target_field)
        if mask is not None:
            mask = reax.metrics.utils.prepare_mask(loss, mask)

        # Now, check for the presence of a user-defined mask
        if self._mask_field:
            user_mask = base.as_array(tree.get_by_path(targets._asdict(), self._mask_field))
            user_mask = reax.metrics.utils.prepare_mask(loss, user_mask)
            if mask is None:
                mask = user_mask
            else:
                mask = mask & user_mask

        graph_mask: Bool[jax.Array, "n_graph ..."] | None = targets.globals.get(keys.MASK)

        root: str = self._target_field[0]
        if root in ("nodes", "edges"):
            segments: Int[Array, "n_graph"] = targets.n_node if root == "nodes" else targets.n_edge

            loss: Float[Array, "n_graph ..."] = graph_ops.segment_reduce(
                loss, segments, reduction=self._reduction, mask=mask, segment_mask=graph_mask
            )

        loss = graph_ops.segment_reduce(
            loss, jnp.array([loss.shape[0]]), reduction=self._reduction, mask=graph_mask
        )
        loss = jnp.mean(loss)

        return loss


class WeightedLoss(GraphLoss):
    """A weighted combination of multiple `GraphLoss` instances.

    The combined loss is ``sum_i weights[i] * loss_fns[i](preds, targets)``.

    Args:
        loss_fns: Sequence of `GraphLoss` instances to combine.
        weights: Sequence of per-term weights. Defaults to all-ones.

    Raises:
        ValueError: If any item in ``loss_fns`` is not a `GraphLoss` or the
            number of weights does not match the number of loss functions.
    """

    _weights: tuple[float, ...]
    _loss_fns: tuple[GraphLoss, ...]

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        loss_fns: Sequence[GraphLoss],
        weights: Sequence[float] | None = None,
    ):
        """
        A weighted combination of multiple graph loss functions.

        This class combines multiple graph loss functions with specified weights to
        create a composite loss function. The weights determine the contribution of
        each individual loss function to the final combined loss.

        Args:
            loss_fns: Sequence of graph loss functions to combine.
            weights: Sequence of weights for each loss function. If None, all
                weights are set to 1.0.

        Raises:
            ValueError: If any element in loss_fns is not a subclass of GraphLoss,
                or if the number of weights does not match the number of loss
                functions.
        """
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
        self._loss_fns = tuple(loss_fns)

    @property
    def weights(self):
        """Per-term loss weights as a gradient-stopped array of shape ``(n_terms,)``."""
        return jax.lax.stop_gradient(jnp.array(self._weights))

    @property
    def reduction(self) -> Literal["sum", "mean"]:
        """The reduction shared by all the terms.

        A weighted sum of terms that were reduced differently isn't one or the other, so in that
        case we fall back to ``"mean"``, which is the reduction all the terms default to.
        """
        reductions = {loss_fn.reduction for loss_fn in self._loss_fns}
        if len(reductions) == 1:
            return next(iter(reductions))

        return "mean"

    def _call(self, predictions: jraph.GraphsTuple, targets: jraph.GraphsTuple) -> jax.Array:
        """Return the weighted sum of the per-term losses."""
        # Calculate the loss for each function
        losses = jnp.array(list(map(lambda loss_fn: loss_fn(predictions, targets), self._loss_fns)))
        return jnp.dot(self.weights, losses)

    def loss_with_contributions(
        self, predictions: jraph.GraphsTuple, target: jraph.GraphsTuple
    ) -> tuple[jax.Array, dict[str, float]]:
        """Return the combined loss plus per-term contributions (keyed by term label)."""
        # Calculate the loss for each function
        losses = jax.array(list(map(lambda loss_fn: loss_fn(predictions, target), self._loss_fns)))
        # Group the contributions into a dictionary keyed by the label
        contribs = dict(zip(list(map(GraphLoss.label, self._loss_fns)), losses))

        return jnp.dot(self.weights, losses), contribs


def _get_pure_loss_fn(loss_fn: str | PureLossFn) -> PureLossFn:
    if isinstance(loss_fn, str):
        return getattr(optax.losses, loss_fn)
    if isinstance(loss_fn, Callable):
        return loss_fn

    raise ValueError(f"Unknown loss function type: {type(loss_fn).__name__}")
