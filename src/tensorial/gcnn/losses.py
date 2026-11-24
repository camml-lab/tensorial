"""Graph-aware loss functions for `jraph.GraphsTuple` model outputs.

Wraps pure (array-level) loss functions with the ability to reduce over batches of
variable-size graphs, optionally apply per-graph weights, and expose per-graph losses
for logging/monitoring.
"""

import abc
from collections.abc import Callable, Sequence
import logging
from typing import TYPE_CHECKING, Final, Literal, cast

import beartype
import jax
import jax.numpy as jnp
import jaxtyping as jt
from jaxtyping import Array, Bool, Float, Int
import jraph
import optax.losses
from pytray import tree
import reax
import reax.pytree
from reax.utils import masking
from typing_extensions import override

from .. import base
from . import _tree, graph_ops, keys, typing, utils

if TYPE_CHECKING:
    from tensorial import gcnn

__all__ = "PureLossFn", "GraphLoss", "WeightedLoss", "Loss"

_LOGGER = logging.getLogger(__name__)

# A pure loss function that doesn't know about graphs, just takes arrays and produces a loss array
PureLossFn = Callable[[jax.Array, jax.Array], jax.Array]


class GraphLoss(reax.pytree.PyTree):
    """Base class for losses over batches of `jraph.GraphsTuple`.

    Subclasses implement `_call` (and optionally `loss_with_contributions`) to
    combine a batch of predictions with a batch of targets into a scalar loss.
    """

    _label: str = reax.pytree.field(static=True)

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
        self, predictions: jraph.GraphsTuple, targets: jraph.GraphsTuple | None = None
    ) -> jax.Array:
        """Return the scalar loss between predictions and targets"""
        if targets is None:
            targets = predictions
        return self._call(predictions, targets)

    def is_active(
        self,
        predictions: jraph.GraphsTuple,  # noqa: ARG002
        targets: jraph.GraphsTuple,  # noqa: ARG002
    ) -> bool:
        """Whether this loss applies to the given batch.

        An optional loss is inactive when the data it needs is missing from the batch, and a
        combined loss (like `WeightedLoss`) leaves it out.  The answer depends only on which
        keys are present, so it is fixed when the loss is traced.
        """
        return True

    @abc.abstractmethod
    def _call(self, predictions: jraph.GraphsTuple, targets: jraph.GraphsTuple) -> jax.Array:
        """Return the scalar loss between predictions and targets"""


class Loss(GraphLoss):
    """Simple loss function that passes values from the graph to a function taking numerical values
    such as optax losses
    """

    _loss_fn: PureLossFn = reax.pytree.field(static=True)
    _target_field: "gcnn.typing.TreePath" = reax.pytree.field(static=True)
    _prediction_field: "gcnn.typing.TreePath" = reax.pytree.field(static=True)
    _mask_field: "gcnn.typing.TreePath | None" = reax.pytree.field(static=True)
    _reduction: Literal["sum", "mean"] = reax.pytree.field(static=True)
    _optional: bool = reax.pytree.field(static=True)
    _normalise_by: "gcnn.typing.TreePath | None" = reax.pytree.field(static=True)

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        loss_fn: str | PureLossFn,
        targets: str,
        predictions: str | None = None,
        *,
        reduction: Literal["sum", "mean"] = "mean",
        label: str | None = None,
        mask_field: str | None = None,
        optional: bool = False,
        normalise_by: str | None = None,
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
            mask_field: The path to an additional mask field in the data structure.
                Padding masks, and the per-field mask next to the target (see
                :func:`~tensorial.gcnn.keys.mask_for`), are always applied when present.
            optional: If ``True``, this loss is left out of a `WeightedLoss` for batches
                that have no target (or prediction) field, e.g. a stress loss on a
                dataset of molecules.  If ``False``, a missing field is an error.
            normalise_by: The path to a per-graph value to divide both the target and
                the prediction by before applying the loss, e.g. ``"n_node"`` for a
                per-atom energy loss.  Only valid for global (per-graph) targets.  Padding
                graphs are excluded, and a node mask inside a real graph does not change its
                ``n_node`` (a structure's energy still includes every atom).

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
        self._optional = optional
        if normalise_by is not None and self._target_field[0] != "globals":
            raise ValueError(
                "normalise_by divides each graph's value by a per-graph quantity, so it can "
                f"only be used with a global target, got '{targets}'"
            )
        self._normalise_by = utils.path_from_str(normalise_by) if normalise_by is not None else None
        super().__init__(label or utils.path_to_str(self._prediction_field))

    @property
    def reduction(self) -> Literal["sum", "mean"]:
        """How per-graph losses are combined into the batch scalar."""
        return self._reduction

    @property
    def optional(self) -> bool:
        """Whether this loss is skipped for batches that lack its fields."""
        return self._optional

    @override
    def is_active(self, predictions: jraph.GraphsTuple, targets: jraph.GraphsTuple) -> bool:
        if not self._optional:
            return True
        return _tree.has_path(targets, self._target_field) and _tree.has_path(
            predictions, self._prediction_field
        )

    def _call(self, predictions: jraph.GraphsTuple, targets: jraph.GraphsTuple) -> jax.Array:
        """Compute the scalar loss by applying the wrapped loss function to the fields."""
        if not self.is_active(predictions, targets):
            raise ValueError(
                f"The optional loss '{self.label()}' has no data in this batch (missing "
                f"'{utils.path_to_str(self._target_field)}' or "
                f"'{utils.path_to_str(self._prediction_field)}').  Optional losses can only be "
                f"skipped when combined with others in a WeightedLoss."
            )

        predictions_dict = predictions._asdict()

        pred_values = base.as_array(tree.get_by_path(predictions_dict, self._prediction_field))
        target_values = base.as_array(tree.get_by_path(targets._asdict(), self._target_field))

        if self._normalise_by is not None:
            pred_values = self._normalise(pred_values, predictions)
            target_values = self._normalise(target_values, targets)

        loss = self._loss_fn(pred_values, target_values)

        # If there is a mask in the graph, then use it by default
        mask = _tree.get_mask(targets, self._target_field)
        if mask is not None:
            mask = masking.prepare_mask(loss, mask)

        # Leave out entries that only hold a filled-in placeholder (e.g. zero stress)
        field_mask = _tree.get_field_mask(targets, self._target_field)
        if field_mask is not None:
            field_mask = masking.prepare_mask(loss, base.as_array(field_mask))
            mask = field_mask if mask is None else mask & field_mask

        # Now, check for the presence of a user-defined mask
        if self._mask_field:
            user_mask = base.as_array(tree.get_by_path(targets._asdict(), self._mask_field))
            user_mask = masking.prepare_mask(loss, user_mask)
            if mask is None:
                mask = user_mask
            else:
                mask = mask & user_mask

        globals_dict = cast("dict[str, Any]", targets.globals)
        graph_mask: Bool[jax.Array, "n_graph ..."] | None = globals_dict.get(keys.MASK)

        root: str = self._target_field[0]
        if root in ("nodes", "edges"):
            segments: Int[Array, "n_graph"] = targets.n_node if root == "nodes" else targets.n_edge

            loss: Float[Array, "n_graph ..."] = graph_ops.segment_reduce(
                loss,
                segments,
                reduction=self._reduction,
                mask=mask,
                segment_mask=graph_mask,
            )
            final_mask = graph_mask
        else:
            # A global target has one entry per graph, so its masks select graphs directly
            final_mask = mask
            if graph_mask is not None:
                graph_mask = masking.prepare_mask(loss, graph_mask)
                final_mask = graph_mask if final_mask is None else final_mask & graph_mask

        loss = graph_ops.segment_reduce(
            loss, jnp.array([loss.shape[0]]), reduction=self._reduction, mask=final_mask
        )
        loss = jnp.mean(loss)

        return loss

    def _normalise(self, values: jax.Array, graph: jraph.GraphsTuple) -> jax.Array:
        """Divide per-graph ``values`` by the graph's ``normalise_by`` value (e.g. its size)."""
        denom = base.as_array(tree.get_by_path(graph._asdict(), self._normalise_by))
        # Padding graphs can be empty, so never divide by zero (they are masked out anyway)
        where = denom != 0
        graph_mask = cast("dict[str, Any]", graph.globals).get(keys.MASK)
        if graph_mask is not None:
            where = where & graph_mask.reshape(where.shape)
        # Divide entry-wise, whatever the trailing shape of the per-graph values
        denom = jnp.where(where, denom, 1).reshape((-1,) + (1,) * (values.ndim - 1))
        return values / denom


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

    _weights: tuple[float, ...] = reax.pytree.field(static=True)
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
        elif len(weights) != len(loss_fns):
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

    @override
    def is_active(self, predictions: jraph.GraphsTuple, targets: jraph.GraphsTuple) -> bool:
        return any(loss_fn.is_active(predictions, targets) for loss_fn in self._loss_fns)

    def _call(self, predictions: jraph.GraphsTuple, targets: jraph.GraphsTuple) -> jax.Array:
        """Return the weighted sum of the per-term losses."""
        loss, _ = self.loss_with_contributions(predictions, targets)
        return loss

    def loss_with_contributions(
        self, predictions: jraph.GraphsTuple, target: jraph.GraphsTuple
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        """Return the combined loss plus per-term contributions (keyed by term label).

        Optional terms whose data is missing from the batch are left out.
        """
        active = [
            (weight, loss_fn)
            for weight, loss_fn in zip(self._weights, self._loss_fns)
            if loss_fn.is_active(predictions, target)
        ]
        if not active:
            raise ValueError(
                "None of the loss terms have data in this batch: "
                f"{[loss_fn.label() for loss_fn in self._loss_fns]}"
            )
        _log_active_terms(self._loss_fns, [loss_fn for _, loss_fn in active])

        weights = jax.lax.stop_gradient(jnp.array([weight for weight, _ in active]))
        losses = jnp.array([loss_fn(predictions, target) for _, loss_fn in active])
        contribs = dict(zip([loss_fn.label() for _, loss_fn in active], losses))

        return jnp.dot(weights, losses), contribs


_LOGGED_TERMS: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()


def _log_active_terms(all_terms: Sequence[GraphLoss], active: Sequence[GraphLoss]) -> None:
    """Log (once) which optional loss terms were left out, so it is never silent."""
    if len(active) == len(all_terms):
        return
    key = (tuple(t.label() for t in all_terms), tuple(t.label() for t in active))
    if key not in _LOGGED_TERMS:
        _LOGGED_TERMS.add(key)
        skipped = [t.label() for t in all_terms if not any(t is a for a in active)]
        _LOGGER.info("Loss terms without data were left out: %s", skipped)


def _get_pure_loss_fn(loss_fn: str | PureLossFn) -> PureLossFn:
    if isinstance(loss_fn, str):
        return getattr(optax.losses, loss_fn)
    if isinstance(loss_fn, Callable):
        return loss_fn

    raise ValueError(f"Unknown loss function type: {type(loss_fn).__name__}")
