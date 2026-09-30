from collections.abc import Callable, Sequence
from typing import Any, Final, TypedDict, TypeVar, cast

import beartype
import equinox as eqx
from flax import linen
import jax
import jaxtyping as jt
import jraph
import optax
import reax
import reax.utils
from typing_extensions import NotRequired, override

from ..gcnn.data import _graph_padding

__all__ = ("ReaxModule",)

OutputT_co = TypeVar("OutputT_co", covariant=True)
InputT = TypeVar("InputT")

MetricsDict = dict[str, reax.Metric | str]
LossFn = Callable[[OutputT_co, InputT], jax.Array]
Optimizer = optax.GradientTransformation | Callable[[], optax.GradientTransformation]

# What to assume about a loss function that doesn't say how it reduces over the batch.  Loss
# functions conventionally return an average, and this is also `reax`'s default for raw logged
# values.
_DEFAULT_LOSS_REDUCTION: Final = "mean"


class StepOutput(TypedDict):
    loss: NotRequired[jt.Array]
    targets: NotRequired[OutputT_co]
    predictions: NotRequired[OutputT_co]


class ReaxModule(reax.Module[InputT, OutputT_co]):
    """Tensorial REAX module."""

    # pylint: disable=method-hidden

    _model: linen.Module | None = None
    _loss_fn: LossFn
    _metrics: reax.metrics.MetricCollection | None = None
    _optimizer: Optimizer

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        model: linen.Module,
        loss_fn: LossFn,
        optimizer: Optimizer,
        scheduler: optax.Schedule | None = None,
        metrics: MetricsDict | None = None,
        jit=True,
        donate_graph=False,
        output: Sequence[str] | None = ("predictions", "targets"),
        loss_reduction: "reax.types.ReduceFn | None" = None,
    ):
        """
        Args:
            loss_reduction: how ``loss_fn`` reduces its value over the batch, which decides how the
                logged losses are combined at the end of an epoch: per-batch averages are averaged,
                per-batch totals are added.  Leave as ``None`` to ask the loss function itself
                (`tensorial.gcnn.GraphLoss` knows), which falls back to ``"mean"`` for plain
                callables that can't be asked.
        """
        super().__init__()
        # Params
        self._metrics: Final[reax.metrics.MetricCollection | None] = (
            metrics if metrics is None else reax.metrics.build_collection(metrics)
        )
        self._output: Final[tuple[str, ...]] = self._init_output(output)
        self._loss_fn: Final[LossFn] = loss_fn
        self._loss_reduction: Final[reax.types.ReduceFn | None] = loss_reduction
        self._model: Final[linen.Module] = model

        # State
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._debug: bool = False
        if jit:
            if donate_graph:
                self.step = eqx.filter_jit(donate="all-except-first")(self.step)
            else:
                self.step = eqx.filter_jit(self.step)

        self.calculate_metrics = eqx.filter_jit(donate="all")(self.calculate_metrics)
        self._forward = eqx.filter_jit(donate="all")(self._forward)

    @staticmethod
    def _init_output(output) -> tuple[str]:
        if output is None:
            return tuple()

        if isinstance(output, str):
            return (output,)

        if isinstance(output, Sequence):
            return tuple(output)

        raise TypeError(f"Unsupported output type: {type(output).__name__}")

    @property
    def debug(self) -> bool:
        return self._debug

    @debug.setter
    def debug(self, value: bool) -> None:
        self._debug = value

    @override
    def configure_model(self, _stage: reax.Stage, batch, /):
        if self.parameters() is None:
            inputs = batch
            if isinstance(batch, tuple):
                inputs = batch[0]

            params = self._model.init(self.rngs(), inputs)
            self.set_parameters(params)

    @override
    def configure_optimizers(self):
        if self.parameters() is None:
            raise reax.exceptions.MisconfigurationException("Model parameters have not been set.")

        if not isinstance(self._optimizer, optax.GradientTransformation):
            if self._scheduler is None:
                # Instantiate the optimizer
                self._optimizer = self._optimizer()

            else:
                # Assume the scheduler can be used as a learning rate function
                self._optimizer = self._optimizer(learning_rate=self._scheduler)

        state = self._optimizer.init(self.parameters())
        return self._optimizer, state

    @override
    def training_step(self, batch: tuple[InputT, OutputT_co], _batch_idx: int, /) -> StepOutput:
        inputs, targets = self._prep_batch(batch)
        batch_size = _get_batch_size(inputs)

        (loss, outs), grads = jax.value_and_grad(self.step, argnums=0, has_aux=True)(
            self.parameters(),
            inputs,
            None,
            self._model.apply,
            self._loss_fn,
            self._metrics,
            self._output,
        )

        # Metrics
        metrics = outs.get("metrics")
        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=metrics is None,
            batch_size=batch_size,
            reduce_fn=self._reduce_fn(),
        )

        if metrics is not None:
            metrics = cast(dict[str, reax.Metric], metrics)
            for name, metric in metrics.items():
                self.log(
                    f"train/{name}",
                    metric,
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    prog_bar=True,
                    batch_size=batch_size,
                )

        step_out = {"loss": loss, "grad": grads}
        if "targets" in self._output:
            step_out["targets"] = targets
        if "predictions" in outs:
            step_out["predictions"] = outs["predictions"]

        return step_out

    @override
    def validation_step(
        self, batch: tuple[InputT, OutputT_co], _batch_idx: int, /
    ) -> StepOutput | None:
        inputs, targets = self._prep_batch(batch)
        batch_size = _get_batch_size(inputs)

        loss, outs = self.step(
            self.parameters(),
            inputs,
            None,
            self._model.apply,
            self._loss_fn,
            self._metrics,
            self._output,
        )

        # Metrics
        metrics = outs.get("metrics")
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=metrics is None,
            batch_size=batch_size,
            reduce_fn=self._reduce_fn(),
        )

        if metrics is not None:
            metrics = cast(reax.metrics.MetricCollection, metrics)
            for name, metric in metrics.items():
                self.log(
                    f"val/{name}",
                    metric,
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    prog_bar=True,
                    batch_size=batch_size,
                )

        if not self._output:
            return None  # No outputs

        step_out = {}
        if "targets" in self._output:
            step_out["targets"] = targets
        if "predictions" in self._output:
            step_out["predictions"] = outs["predictions"]

        return step_out

    @override
    def test_step(self, batch: tuple[InputT, OutputT_co], _batch_idx: int, /) -> StepOutput | None:
        inputs, targets = self._prep_batch(batch)
        batch_size = _get_batch_size(inputs)

        loss, outs = self.step(
            self.parameters(),
            inputs,
            None,
            self._model.apply,
            self._loss_fn,
            self._metrics,
            self._output,
        )

        # Metrics
        metrics = outs.get("metrics")
        self.log(
            "test/loss",
            loss,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=metrics is None,
            batch_size=batch_size,
            reduce_fn=self._reduce_fn(),
        )

        if metrics is not None:
            metrics = cast(reax.metrics.MetricCollection, metrics)
            for name, metric in metrics.items():
                self.log(
                    f"test/{name}",
                    metric,
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    prog_bar=True,
                    batch_size=batch_size,
                )

        if not self._output:
            return None  # No outputs

        step_out = {}
        if "targets" in self._output:
            step_out["targets"] = targets
        if "predictions" in self._output:
            step_out["predictions"] = outs["predictions"]

        return step_out

    @override
    def predict_step(self, batch: InputT, _batch_idx: int, /) -> OutputT_co:
        inputs, _outputs = self._prep_batch(batch)
        return self._forward(self.parameters(), inputs, self._model.apply)

    @staticmethod
    def _forward(
        params: jt.PyTree, inputs: InputT, model: Callable[[jt.PyTree, InputT], OutputT_co]
    ) -> OutputT_co:
        return model(params, inputs)

    @staticmethod
    def step(
        params: jt.PyTree,
        inputs: InputT,
        _targets: OutputT_co,
        model: Callable[[jt.PyTree, InputT], OutputT_co],
        loss_fn: LossFn,
        metrics: reax.metrics.MetricCollection | None = None,
        output: tuple[str, ...] = tuple(),
    ) -> tuple[jax.Array, dict]:
        """Calculate loss and, optionally metrics."""
        outs = {}
        predictions = model(params, inputs)
        if "predictions" in output:
            outs["predictions"] = predictions

        if metrics:
            metrics = metrics.create(predictions, inputs)
            outs["metrics"] = metrics

        return loss_fn(predictions, inputs), outs

    @override
    def on_before_optimizer_step(self, _optimizer: reax.Optimizer, grad: dict[str, Any], /):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        if self.debug and self.trainer.current_epoch % 25 == 0:
            norms = reax.utils.grad_norm(grad, norm_type=2)
            self.log_dict(norms, on_step=False, on_epoch=True, logger=True, prog_bar=False)

    @staticmethod
    def calculate_metrics(
        predictions: OutputT_co, targets: OutputT_co, metrics: MetricsDict
    ) -> dict[str, reax.Metric]:
        return {key: metric.create(predictions, targets) for key, metric in metrics.items()}

    def _reduce_fn(self) -> "reax.types.ReduceFn":
        """How the losses logged over an epoch should be reduced to a single value.

        This follows the reduction the loss function itself applies over each batch: per-batch
        averages are averaged, per-batch totals are added, so the epoch value stays on the scale the
        loss was asked to produce.  `tensorial.gcnn.GraphLoss` says which it does, otherwise we fall
        back to whatever was passed as ``loss_reduction``.
        """
        if self._loss_reduction is not None:
            return self._loss_reduction

        return getattr(self._loss_fn, "reduction", _DEFAULT_LOSS_REDUCTION)

    def _prep_batch(self, batch) -> tuple[InputT, OutputT_co | None]:
        if isinstance(batch, jraph.GraphsTuple):
            inputs = outputs = batch
        else:
            if len(batch) == 1:
                inputs = outputs = batch[0]
            else:
                inputs, outputs = batch

        return inputs, outputs


def _get_batch_size(inputs: InputT):
    if not isinstance(inputs, jraph.GraphsTuple):
        return None

    try:
        mask = inputs.globals["mask"]
    except KeyError:
        mask = _graph_padding.get_graph_padding_mask(inputs)

    return mask.sum()
