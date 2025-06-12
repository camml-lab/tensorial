from collections.abc import Callable
from typing import Any, Final, Optional, Union, cast

import beartype
import equinox as eqx
from flax import linen
import jax
import jaxtyping as jt
import jraph
import optax
import reax
import reax.utils
from typing_extensions import override

__all__ = ("ReaxModule",)

MetricsDict = dict[str, Union[reax.Metric, str]]
LossFn = Callable[[jraph.GraphsTuple, jraph.GraphsTuple], jax.Array]
Optimizer = Union[optax.GradientTransformation, Callable[[], optax.GradientTransformation]]


class ReaxModule(reax.Module[jraph.GraphsTuple, jraph.GraphsTuple]):
    """Tensorial REAX module."""

    # pylint: disable=method-hidden

    _model: Optional[linen.Module] = None
    _loss_fn: LossFn
    _metrics: Optional[reax.metrics.MetricCollection] = None
    _optimizer: Optimizer

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        model: linen.Module,
        loss_fn: LossFn,
        optimizer: Optimizer,
        scheduler: Optional[optax.Schedule] = None,
        metrics: Optional[MetricsDict] = None,
        jit=True,
    ):
        super().__init__()
        self._model = model
        self._loss_fn = loss_fn
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._metrics: Final[Optional[reax.metrics.MetricCollection]] = (
            metrics if metrics is None else reax.metrics.build_collection(metrics)
        )
        self._debug = False
        if jit:
            self.step = eqx.filter_jit(donate="all-except-first")(self.step)
            self.calculate_metrics = eqx.filter_jit(donate="all")(self.calculate_metrics)
            self._forward = eqx.filter_jit(donate="all")(self._forward)

    @property
    def debug(self) -> bool:
        return self._debug

    @debug.setter
    def debug(self, value: bool) -> None:
        self._debug = value

    @override
    def configure_model(self, stage: reax.Stage, batch, /):
        if self.parameters() is None:
            inputs = batch
            if isinstance(batch, tuple):
                inputs = batch[0]

            params = self._model.init(self.rng_key(), inputs)
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
    def training_step(
        self, batch: tuple[jraph.GraphsTuple, jraph.GraphsTuple], batch_idx: int, /
    ) -> tuple[jax.Array, jax.Array]:
        inputs, outputs = self._prep_batch(batch)
        (loss, metrics), grads = jax.value_and_grad(self.step, argnums=0, has_aux=True)(
            self.parameters(), inputs, outputs, self._model.apply, self._loss_fn, self._metrics
        )
        have_metrics = metrics is not None
        self.log(
            "train/loss", loss, on_step=False, on_epoch=True, logger=True, prog_bar=not have_metrics
        )

        if metrics:
            metrics = cast(dict[str, reax.Metric], metrics)
            for name, metric in metrics.items():
                self.log(
                    f"train/{name}",
                    metric,
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    prog_bar=True,
                )

        return loss, grads

    @override
    def validation_step(
        self, batch: tuple[jraph.GraphsTuple, jraph.GraphsTuple], batch_idx: int, /
    ):
        inputs, outputs = self._prep_batch(batch)
        loss, metrics = self.step(
            self.parameters(), inputs, outputs, self._model.apply, self._loss_fn, self._metrics
        )
        have_metrics = metrics is not None
        self.log(
            "val/loss", loss, on_step=False, on_epoch=True, logger=True, prog_bar=not have_metrics
        )

        if have_metrics:
            metrics = cast(reax.metrics.MetricCollection, metrics)
            for name, metric in metrics.items():
                self.log(
                    f"val/{name}",
                    metric,
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    prog_bar=True,
                )

    @override
    def predict_step(self, batch: jraph.GraphsTuple, batch_idx: int, /) -> jraph.GraphsTuple:
        inputs, _outputs = self._prep_batch(batch)
        return self._forward(self.parameters(), inputs, self._model.apply)

    @staticmethod
    def _forward(
        params: jt.PyTree,
        inputs: jraph.GraphsTuple,
        model: Callable[[jt.PyTree, jraph.GraphsTuple], jraph.GraphsTuple],
    ) -> jraph.GraphsTuple:
        return model(params, inputs)

    @staticmethod
    def step(
        params: jt.PyTree,
        inputs: jraph.GraphsTuple,
        _targets: jraph.GraphsTuple,
        model: Callable[[jt.PyTree, jraph.GraphsTuple], jraph.GraphsTuple],
        loss_fn: Callable,
        metrics: Optional[reax.metrics.MetricCollection] = None,
    ) -> tuple[jax.Array, Optional[reax.metrics.MetricCollection]]:
        """Calculate loss and, optionally metrics."""
        predictions = model(params, inputs)
        if metrics:
            metrics = metrics.create(predictions, inputs)

        return loss_fn(predictions, inputs), metrics

    @override
    def on_before_optimizer_step(self, optimizer: "reax.Optimizer", grad: dict[str, Any], /):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        if self.debug and self.trainer.current_epoch % 25 == 0:
            norms = reax.utils.grad_norm(grad, norm_type=2)
            self.log_dict(norms, on_step=False, on_epoch=True, logger=True, prog_bar=False)

    @staticmethod
    def calculate_metrics(
        predictions: jraph.GraphsTuple, targets: jraph.GraphsTuple, metrics: MetricsDict
    ) -> dict[str, reax.Metric]:
        return {key: metric.create(predictions, targets) for key, metric in metrics.items()}

    def _prep_batch(self, batch) -> tuple[jraph.GraphsTuple, Optional[jraph.GraphsTuple]]:
        if isinstance(batch, jraph.GraphsTuple):
            inputs = outputs = batch
        else:
            if len(batch) == 1:
                inputs = outputs = batch[0]
            else:
                inputs, outputs = batch

        return inputs, outputs
