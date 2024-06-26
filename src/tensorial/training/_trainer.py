from collections.abc import Callable
import itertools
from typing import Any, Final, Generic, Optional, TypeVar
import uuid

import beartype
import clu.metrics
from flax.training import train_state
import jax
import jax.numpy as jnp
import jaxtyping as jt
import optax

from tensorial import data, training
import tensorial.metrics

__all__ = (
    "Trainer",
    "DEFAULT_MAX_EPOCHS",
    "TRAIN_MAX_EPOCHS",
    "Batch",
    "ModelT",
    "JIT_ALL",
    "JIT_EVAL",
    "JIT_TRAIN",
)

PyTree = Any
InputT_co = TypeVar("InputT_co", covariant=True)
OutputT_co = TypeVar("OutputT_co", covariant=True)
LabelT_co = TypeVar("LabelT_co", covariant=True)
Batch = tuple[InputT_co, LabelT_co]
ModelT = Callable[[PyTree, InputT_co], OutputT_co]
LossFn = Callable[[OutputT_co, LabelT_co], jax.Array]

TRAIN_MAX_EPOCHS = "max_epochs"
TRAIN_STOP = "stop"

DEFAULT_MAX_EPOCHS: Final[int] = 10_000
DEFAULT_OVERFITTING_WINDOW = 50

JIT_TRAIN = 0b001
JIT_EVAL = 0b010
JIT_ALL = JIT_TRAIN | JIT_EVAL

DefaultMetrics = clu.metrics.Collection.create(loss=clu.metrics.Average.from_output("loss"))


class Trainer(Generic[InputT_co, OutputT_co]):
    """Simple trainer with some convenience functionality built in"""

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
            self,
            model: ModelT[InputT_co, OutputT_co],
            model_params: PyTree,
            opt: optax.GradientTransformation,
            loss_fn: LossFn[OutputT_co, LabelT_co],
            train_data: data.DataLoader[Batch],
            validate_data: data.DataLoader[Batch] = None,
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
        self._metrics = metrics or DefaultMetrics

        self._logger = training.TrainingLogger(log_metrics_every)
        self.add_listener(self._logger)
        self._overfitting = training.EarlyStopping(overfitting_window)

        self._train_step = train_step
        # Use bitmask to see if the users wants to jit calls to the model
        if jit & JIT_TRAIN:
            self._train_step = jax.jit(train_step, static_argnums=2)

        eval_fn = eval_step
        if jit & JIT_EVAL:
            eval_fn = jax.jit(eval_step, static_argnums=[1, 3])
        self._evaluator = tensorial.metrics.Evaluator(self._metrics, eval_fn)

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
                metrics = self._metrics.empty()
                state = self._train_state
                for batch in self._train_data:
                    state, _loss, metrics = self._train_step(
                        state, batch, loss_fn=self._loss_fn, metrics=metrics
                    )
                self._train_metrics = metrics.compute()

                # Now update out state
                self._train_state = state

                if self._validate_data is not None:
                    # Now do validation pass
                    self._validate_metrics = self._evaluator.evaluate(
                        self._validate_data, state=self._train_state, loss_fn=self._loss_fn
                    )

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


@jt.jaxtyped(typechecker=beartype.beartype)
def train_step(
        state: train_state.TrainState,
        batch: Batch[InputT_co, LabelT_co],
        loss_fn: LossFn[OutputT_co, LabelT_co],
        metrics: clu.metrics.Collection,
) -> tuple[train_state.TrainState, Any, clu.metrics.Collection]:
    """Train for a single step."""
    inputs, labels = batch
    if labels is None:
        labels = inputs

    def loss_fn_shim(params):
        """
        Shim to have a function that takes parameters of the model and returns the loss.  This makes
        it possible to call grad to get derivatives.
        """
        outputs = state.apply_fn(params, inputs)
        return loss_fn(outputs, labels), outputs

    grad_fn = jax.value_and_grad(loss_fn_shim, has_aux=True)
    (loss, predictions), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)

    update = metrics.single_from_model_output(
        loss=jnp.astype(loss, jnp.float32),
        values=predictions,
        targets=labels,
    )
    return state, loss, metrics.merge(update)


@jt.jaxtyped(typechecker=beartype.beartype)
def eval_step(
        batch: Batch[InputT_co, LabelT_co],
        metrics: type[clu.metrics.Collection],
        state: train_state.TrainState,
        loss_fn: LossFn[OutputT_co, LabelT_co],
) -> clu.metrics.Collection:
    """Evaluate for a single step."""
    inputs, labels = batch
    if labels is None:
        labels = inputs

    outputs = state.apply_fn(state.params, inputs)
    loss = loss_fn(outputs, labels)
    update = metrics.single_from_model_output(
        loss=jnp.astype(loss, jnp.float32), values=outputs, targets=labels
    )
    return update
