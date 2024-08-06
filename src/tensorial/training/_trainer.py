from collections.abc import Callable
import itertools
from typing import Any, Final, Generic, Optional, TypeVar
import uuid

import beartype
from flax.training import train_state
import jax
import jaxtyping as jt
import optax

import tensorial
from tensorial import data, training

from . import _steps

__all__ = (
    "Trainer",
    "DEFAULT_MAX_EPOCHS",
    "TRAIN_MAX_EPOCHS",
    "BatchT",
    "ModelT",
    "JIT_ALL",
    "JIT_EVAL",
    "JIT_TRAIN",
)

PyTree = Any
InputT_co = TypeVar("InputT_co", covariant=True)
OutputT_co = TypeVar("OutputT_co", covariant=True)
LabelT_co = TypeVar("LabelT_co", covariant=True)
BatchT = TypeVar("BatchT")
ModelT = Callable[[PyTree, InputT_co], OutputT_co]
LossFn = Callable[[OutputT_co, LabelT_co], jax.Array]

TRAIN_MAX_EPOCHS = "max_epochs"
TRAIN_STOP = "stop"

DEFAULT_MAX_EPOCHS: Final[int] = 10_000
DEFAULT_OVERFITTING_WINDOW = 50

JIT_TRAIN = 0b001
JIT_EVAL = 0b010
JIT_ALL = JIT_TRAIN | JIT_EVAL


class Trainer(Generic[InputT_co, OutputT_co]):
    """Simple trainer with some convenience functionality built in"""

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        model: ModelT[InputT_co, OutputT_co],
        model_params: PyTree,
        opt: optax.GradientTransformation,
        loss_fn: LossFn[OutputT_co, LabelT_co],
        train_data: data.DataLoader[BatchT],
        validate_data: data.DataLoader[BatchT] = None,
        metrics: tensorial.metrics.MetricCollection = None,
        log_metrics_every=1,
        overfitting_window=DEFAULT_OVERFITTING_WINDOW,
        jit=JIT_ALL,
    ):
        super().__init__()
        self._model = model
        self._opt = opt
        self._loss_fn = loss_fn
        self._train_data = train_data
        self._validate_data = validate_data

        self._events = training.EventGenerator()
        self._stopping = False
        self._stop_msg = None
        self._train_metrics = None
        self._validate_metrics = None

        # Track all the losses
        self._epoch = 0

        self._train_state = train_state.TrainState.create(
            apply_fn=model, params=model_params, tx=opt
        )
        self._logger = training.TrainingLogger(log_metrics_every)
        self.add_listener(self._logger)
        self._overfitting = training.EarlyStopping(overfitting_window)

        self._steps = _steps.SimpleModule(self._loss_fn, metrics)

        self._train_step = jax.grad(self._steps.training_step, argnums=0, has_aux=True)
        # Use bitmask to see if the users wants to jit calls to the model
        if jit & JIT_TRAIN:
            self._train_step = jax.jit(self._train_step, static_argnums=1)

        self._eval_step = self._steps.validation_step
        if jit & JIT_EVAL:
            self._eval_step = jax.jit(self._eval_step, static_argnums=1)

    @property
    def train_data(self) -> data.DataLoader:
        return self._train_data

    @property
    def validate_data(self) -> Optional[data.DataLoader]:
        return self._validate_data

    @property
    def epoch(self) -> int:
        """Return the current training epoch"""
        return self._epoch

    @property
    def train_metrics(self) -> Optional[dict]:
        """The metrics from the last training step"""
        return self._train_metrics

    @property
    def validate_metrics(self) -> Optional[dict]:
        """The metrics from the last validation step"""
        return self._validate_metrics

    @property
    def metrics_log(self) -> training.TrainingLogger:
        return self._logger

    def stop(self, reason: str):
        """Stop training at the next opportunity (usually the end of the current epoch)"""
        self._stop_msg = reason
        self._stopping = True

    def add_listener(self, listener: training.TrainerListener) -> uuid.UUID:
        return self._events.add_listener(listener)

    def remove_listener(self, handle):
        return self._events.remove_listener(handle)

    def train(
        self,
        min_epochs: int = None,
        max_epochs=DEFAULT_MAX_EPOCHS,
    ) -> str:
        """
        Train the model by passing the training data through and upgrading the parameters using the
        optimiser. If validation data is available, after each training step the validation set will
        be passed through the model gather metrics along the way.  Note, this means that at each
        step the loss metrics will reflect values averaged over the training batches as the
        gradients are updated after each batch while the validation metrics will reflect the state
        of the model at the end of the epoch.
        """
        self._stopping = False
        iterator = itertools.count() if max_epochs == -1 else range(max_epochs)

        with self._events.listen_context(self._overfitting):
            self._events.fire_event(training.TrainerListener.on_training_starting, self)

            # Loop over epochs
            for local_epoch in iterator:
                epoch = self._epoch
                self._events.fire_event(training.TrainerListener.on_epoch_starting, self, epoch)

                # Iterate over training batches
                metrics = None
                state = self._train_state
                for batch_idx, batch in enumerate(self._train_data):
                    grads, outs = self._train_step(state.params, state.apply_fn, batch)
                    # Update state
                    state = state.apply_gradients(grads=grads)
                    # Update metrics
                    metrics = outs.metric if batch_idx == 0 else metrics.merge(outs.metric)

                self._train_metrics = metrics.compute()

                # Now update out state
                self._train_state = state

                if self._validate_data is not None:
                    # Now do validation pass
                    metrics = None
                    for batch_idx, batch in enumerate(self._train_data):
                        loss, outs = self._eval_step(state.params, state.apply_fn, batch)
                        metrics = outs.metric if batch_idx == 0 else metrics.merge(outs.metric)

                    self._validate_metrics = metrics.compute()

                self._events.fire_event(training.TrainerListener.on_epoch_finishing, self, epoch)
                # Tell everyone that the epoch is finishing
                self._epoch += 1
                # And tell everyone that this epoch is over
                self._events.fire_event(training.TrainerListener.on_epoch_finished, self, epoch)

                if (min_epochs is not None and local_epoch > (min_epochs - 1)) and self._stopping:
                    break

        stop_msg = TRAIN_MAX_EPOCHS if not self._stopping else self._stop_msg

        # Tell everyone that we are stopping
        self._events.fire_event(training.TrainerListener.on_training_stopping, self, stop_msg)

        return stop_msg
