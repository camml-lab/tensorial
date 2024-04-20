# -*- coding: utf-8 -*-
import contextlib
import itertools
import math
from typing import Any, Callable, Iterable, List, Optional, Tuple
import uuid

import clu.metrics
from flax.training import train_state
import jax
import optax

Batch = Tuple[Any, Any]
Dataset = Iterable[Batch]

TRAIN_MAX_EPOCHS = 'max_epochs'
TRAIN_OVERFITTING = 'overfitting'
TRAIN_STOP = 'stop'

DEFAULT_MAX_EPOCHS = 30_000
DEFAULT_OVERFITTING_WINDOW = 50

JIT_TRAIN = 0b001
JIT_EVAL = 0b010
JIT_ALL = JIT_TRAIN | JIT_EVAL


class TrainerListener:

    def on_epoch_starting(self, trainer: 'Trainer', epoch_num: int):
        """A training epoch has started"""

    def on_epoch_finishing(self, trainer: 'Trainer', epoch_num: int):
        """An epoch is finishing but the model state has not been updated yet"""

    def on_epoch_finished(self, trainer: 'Trainer', epoch_num: int):
        """An epoch has finished and the model state has been updated"""

    def on_training_stopping(self, trainer: 'Trainer', stop_msg: str):
        """A training run is stopping"""


class EventGenerator:
    """Manage listeners and fire events"""

    def __init__(self):
        self._event_listeners = {}

    def add_listener(self, listener) -> Any:
        handle = uuid.uuid4()
        self._event_listeners[handle] = listener
        return handle

    def remove_listener(self, handle) -> Any:
        return self._event_listeners.pop(handle)

    def fire_event(self, event_fn: Callable, *args, **kwargs):
        for listener in self._event_listeners.values():
            getattr(listener, event_fn.__name__)(*args, **kwargs)

    @contextlib.contextmanager
    def listen_context(self, *listener: TrainerListener):
        uuids = tuple()
        try:
            uuids = tuple(map(self.add_listener, listener))
            yield
        finally:
            map(self.remove_listener, uuids)


class TrainingLogger(TrainerListener):
    """Log metrics for training and validation"""

    def __init__(self, log_every_n_epochs: int = 1, log_on_stop: bool = True):
        self._log: List[dict] = []
        self._log_ever_n_epochs = log_every_n_epochs
        self._log_on_stop = log_on_stop

    def raw_log(self) -> List[dict]:
        return self._log

    def as_dataframe(self):
        import pandas
        return pandas.DataFrame(self._log)

    def on_epoch_finished(self, trainer: 'Trainer', epoch_num: int):
        if epoch_num % self._log_ever_n_epochs == 0:
            self._save_log(trainer, epoch_num)

    def _save_log(self, trainer: 'Trainer', epoch):
        log_entry = {'epoch': epoch}
        if trainer.train_metrics is not None:
            log_entry.update({'training_' + key: value for key, value in trainer.train_metrics.items()})
        if trainer.validate_metrics is not None:
            log_entry.update({'validation_' + key: value for key, value in trainer.validate_metrics.items()})
        self._log.append(log_entry)


class EarlyStopping(TrainerListener):
    """
    Basic early stopping class.  If, during training, the chosen metric (defaults to loss) increases by `min_delta` for
    `patience` steps in a row then the training is stopped.
    """
    _best_value = float('inf')

    def __init__(self, patience: int, metric='loss', min_delta: float = 0.):
        self._patience = patience
        self._metric = metric
        self._num_increases = 0
        self._min_delta = min_delta
        self._has_improved = False

    @property
    def patience(self) -> int:
        return self._patience

    @property
    def has_improved(self) -> bool:
        return self._has_improved

    def on_epoch_finished(self, trainer: 'Trainer', epoch_num):
        new_value = trainer.validate_metrics[self._metric]
        if math.isinf(self._best_value) or self._best_value - new_value > self._min_delta:
            # Reset
            self._num_increases = 0
            self._best_value = new_value
            self._has_improved = True
        else:
            self._num_increases += 1
            self._has_improved = False

        if self._num_increases > self.patience:
            trainer.stop(TRAIN_OVERFITTING)


DefaultMetrics = clu.metrics.Collection.create(loss=clu.metrics.Average.from_output('loss'))


class Trainer:

    def __init__(
        self,
        model: Callable,
        model_params,
        opt: optax.GradientTransformation,
        loss_fn: Callable,
        train_data: Dataset,
        validate_data: Dataset = None,
        metrics: type[clu.metrics.Collection] = None,
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

        self._events = EventGenerator()
        self._stopping = False
        self._stop_msg = None
        self._train_metrics = None
        self._validate_metrics = None

        # Track all the losses
        self._epoch = 0

        self._train_state = train_state.TrainState.create(apply_fn=model, params=model_params, tx=opt)
        self._metrics = metrics or DefaultMetrics

        self._logger = TrainingLogger(log_metrics_every)
        self.add_listener(self._logger)
        self._overfitting = EarlyStopping(overfitting_window)

        self._train_step = train_step
        self._eval_step = eval_step
        # Use bitmask to see if the users wants to jit calls to the model
        if jit & JIT_TRAIN:
            self._train_step = jax.jit(train_step, static_argnums=2)
        if jit & JIT_EVAL:
            self._eval_step = jax.jit(eval_step, static_argnums=2)

    @property
    def train_data(self) -> Dataset:
        return self._train_data

    @property
    def validate_data(self) -> Optional[Dataset]:
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
    def metrics_log(self) -> TrainingLogger:
        return self._logger

    def stop(self, reason: str):
        """Stop training at the next opportunity (usually the end of the current epoch)"""
        self._stop_msg = reason
        self._stopping = True

    def add_listener(self, listener: TrainerListener) -> uuid.UUID:
        return self._events.add_listener(listener)

    def remove_listener(self, handle):
        return self._events.remove_listener(handle)

    def train(
        self,
        min_epochs: int = None,
        max_epochs=DEFAULT_MAX_EPOCHS,
    ) -> str:
        """
        Train the model by passing the training data through and upgrading the parameters using the optimiser.
        If validation data is available, after each training step the validation set will be passed through the model
        gather metrics along the way.  Note, this means that at each step the loss metrics will reflect values averaged
        over the training batches as the gradients are updated after each batch while the validation metrics will
        reflect the state of the model at the end of the epoch.
        """
        self._stopping = False
        iterator = itertools.count() if max_epochs == -1 else range(max_epochs)

        with self._events.listen_context(self._overfitting):
            # Loop over epochs
            for local_epoch in iterator:
                epoch = self._epoch
                self._events.fire_event(TrainerListener.on_epoch_starting, self, epoch)

                # Iterate over training batches
                metrics = self._metrics.empty()
                state = self._train_state
                for batch in self._train_data:
                    state, _loss, metrics = self._train_step(state, batch, loss_fn=self._loss_fn, metrics=metrics)
                self._train_metrics = metrics.compute()

                # Now update out state
                self._train_state = state

                self._events.fire_event(TrainerListener.on_epoch_finishing, self, epoch)

                if self._validate_data is not None:
                    # Now do validation pass
                    metrics = self._metrics.empty()
                    for batch in self._validate_data:
                        _loss, metrics = self._eval_step(
                            self._train_state, batch, loss_fn=self._loss_fn, metrics=metrics
                        )
                    self._validate_metrics = metrics.compute()

                # Tell everyone that the epoch is finishing
                self._epoch += 1
                # And tell everyone that this epoch is over
                self._events.fire_event(TrainerListener.on_epoch_finished, self, epoch)

                if (min_epochs is not None and local_epoch > (min_epochs - 1)) and self._stopping:
                    break

        stop_msg = TRAIN_MAX_EPOCHS if not self._stopping else self._stop_msg

        # Tell everyone that we are stopping
        self._events.fire_event(TrainerListener.on_training_stopping, self, stop_msg)

        return stop_msg


def train_step(state: train_state.TrainState, batch: Batch, loss_fn: Callable, metrics: clu.metrics.Collection) -> \
        Tuple[train_state.TrainState, Any, clu.metrics.Collection]:
    """Train for a single step."""
    inputs, labels = batch

    def loss_fn_shim(params):
        """Shim to have a function that takes parameters of the model and returns the loss.  This makes it possible to
        call grad to get derivatives."""
        predictions = state.apply_fn(params, inputs)
        return loss_fn(predictions, labels), predictions

    grad_fn = jax.value_and_grad(loss_fn_shim, has_aux=True)
    (loss, predictions), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)

    update = metrics.single_from_model_output(loss=loss, labels=labels, predictions=predictions)
    return state, loss, metrics.merge(update)


def eval_step(state: train_state.TrainState, batch: Batch, loss_fn: Callable,
              metrics: clu.metrics.Collection) -> Tuple[Any, clu.metrics.Collection]:
    """Evaluate for a single step."""
    inputs, labels = batch
    predictions = state.apply_fn(state.params, inputs)
    loss = loss_fn(predictions, labels)
    update = metrics.single_from_model_output(loss=loss, labels=labels, predictions=predictions)
    return loss, metrics.merge(update)
