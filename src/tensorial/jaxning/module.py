import abc
from collections.abc import Callable, Sequence
from typing import Any, Generic, Optional, TypeVar, Union

import beartype
import flax.training.train_state
import jax
import jaxtyping as jt
import optax

from tensorial import metrics as metrics_mod
import tensorial.metrics

from . import exceptions, hooks, rank_zero

__all__ = "Module", "SimpleModule"

MetricType = Union["Metric", jax.typing.ArrayLike]

OutputT_co = TypeVar("OutputT_co", covariant=True)
BatchT = TypeVar("BatchT")

TrainState = flax.training.train_state.TrainState


class Module(Generic[BatchT, OutputT_co], hooks.ModelHooks):
    def __init__(self, rng_key: jax.Array = None):
        self._trainer = None
        self._rng_key = rng_key or jax.random.key(0)
        self._parameters = None
        self._automatic_optimization = True

    @property
    def automatic_optimization(self) -> bool:
        return self._automatic_optimization

    @automatic_optimization.setter
    def automatic_optimization(self, automatic_optimization: bool) -> None:
        self._automatic_optimization = automatic_optimization

    @property
    def trainer(self):
        return self._trainer

    @trainer.setter
    def trainer(self, trainer):
        if self._trainer is not None and trainer is not None:
            raise RuntimeError("Cannot set trainer, it is already set.")

        self._trainer = trainer

    def parameters(self) -> Optional[jt.PyTree]:
        return self._parameters

    def set_parameters(self, params: jt.PyTree):
        self._parameters = params

    def rng_key(self, num=1) -> jax.Array:
        self._rng_key, subkey = jax.random.split(self._rng_key, num=num + 1)
        return subkey

    def optimizers(self) -> Union[optax.GradientTransformation, list[optax.GradientTransformation]]:
        optimizers = self.trainer.optimizers

        # Check for a single optimiser
        if (
            isinstance(optimizers, list)
            and len(optimizers) == 1
            and isinstance(optimizers[0], optax.GradientTransformation)
        ):
            return optimizers[0]

        # Multiple optimisers
        return optimizers

    @abc.abstractmethod
    def training_step(
        self, state: flax.training.train_state.TrainState, batch: BatchT
    ) -> Optional[tuple[jax.Array, jax.Array]]:
        """Train step"""

    @abc.abstractmethod
    def validation_step(
        self, state: TrainState, batch: BatchT
    ) -> tensorial.metrics.MetricCollection:
        """Validate step"""

    def configure_optimizers(
        self,
    ) -> Union[optax.GradientTransformation, Sequence[optax.GradientTransformation], None]:
        """Create the optimizer(s) to use during training"""
        return None

    def log(
        self,
        name: str,
        value: MetricType,
        prog_bar: bool = False,
        batch_size: Optional[int] = None,
        logger: Optional[bool] = None,
        on_step=True,
        on_epoch=True,
    ) -> None:
        """Log a key, value pair.

        Example::

            self.log('train_loss', loss)

        """
        trainer = self._trainer
        if trainer is None:
            # not an error to support testing the `*_step` methods without a `Trainer` reference
            rank_zero.warn(
                "`self.log()` was called before `self.trainer` was set. "
                "Probably, the model was not passed to `Trainer`"
            )
            return

        if logger and trainer.logger is None:
            rank_zero.warn(
                f"You called `self.log({name!r}, ..., logger=True)` but have no logger configured. You can enable one"
                " by doing `Trainer(logger=ALogger(...))`"
            )
        if logger is None:
            # we could set false here if there's no configured logger, however, we still need to compute the "logged"
            # metrics anyway because that's what the evaluation loops use as return value
            logger = True

        trainer.log(
            name,
            value,
            prog_bar=prog_bar,
            batch_size=batch_size,
            logger=logger,
            on_step=True,
            on_epoch=True,
        )


PyTree = Any
InputT_co = TypeVar("InputT_co", covariant=True)
ModelT = Callable[[PyTree, InputT_co], OutputT_co]
LabelT_co = TypeVar("LabelT_co", covariant=True)
LossFn = Callable[[OutputT_co, LabelT_co], jax.Array]


class SimpleModule(Module[InputT_co, OutputT_co]):
    _loss_fn: LossFn[OutputT_co, LabelT_co]
    _metrics: metrics_mod.MetricCollection

    def __init__(
        self, loss_fn: LossFn[OutputT_co, LabelT_co], metrics: metrics_mod.MetricCollection
    ):
        super().__init__()
        self._loss_fn = loss_fn
        self._metrics = metrics

    @jt.jaxtyped(typechecker=beartype.beartype)
    def training_step(self, state: flax.training.train_state.TrainState, batch: BatchT):
        """Train for a single step."""
        state, metrics = self.train(state, self._loss_fn, self._metrics, batch)
        # TODO: Log metrics
        # Update the state
        self._state = state

    @jt.jaxtyped(typechecker=beartype.beartype)
    def validation_step(
        self, state: flax.training.train_state.TrainState, batch: BatchT
    ) -> tensorial.metrics.MetricCollection:
        """Validate for a single step."""
        inputs, labels = batch
        if labels is None:
            labels = inputs

        predictions = state.apply_fn(state.params, inputs)
        loss = self._loss_fn(predictions, labels)
        metrics = self._metrics.create(predictions, labels)

        return metrics

    @staticmethod
    @jax.jit
    def train(
        state: TrainState, loss_fn: Callable, metrics, batch: BatchT
    ) -> tuple[TrainState, Any]:
        inputs, labels = batch
        if labels is None:
            labels = inputs

        def loss_shim(params):
            outputs = state.apply_fn(params, inputs)
            return loss_fn(outputs, labels), outputs

        grad_fn = jax.value_and_grad(loss_shim, has_aux=True)
        predictions, grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        metrics = metrics.create(predictions, labels)
        # TODO: Add average loss to metrics

        return state, metrics

    @staticmethod
    @jt.jaxtyped(typechecker=beartype.beartype)
    def evaluate(
        state: TrainState, loss_fn: Callable, metrics, batch: BatchT
    ) -> tuple[jax.Array, tensorial.metrics.MetricCollection]:
        """Train for a single step."""
        inputs, labels = batch
        if labels is None:
            labels = inputs

        outputs = state.apply_fn(state.params, inputs)
        loss = loss_fn(outputs, labels), outputs

        metrics = metrics.create(outputs, labels)
        # TODO: Add average loss to metrics

        return loss, metrics


import lightning

optax.GradientTransformation

lightning.Trainer
