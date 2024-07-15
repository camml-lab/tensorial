import abc
from collections.abc import Callable
from typing import Any, Generic, Optional, Type, TypeVar

import beartype
import clu.metrics
import equinox
import flax.struct
import jax
import jax.numpy as jnp
import jaxtyping as jt

PyTree = Any
InputT_co = TypeVar("InputT_co", covariant=True)
OutputT_co = TypeVar("OutputT_co", covariant=True)
LabelT_co = TypeVar("LabelT_co", covariant=True)
Batch = tuple[InputT_co, LabelT_co]
ModelT = Callable[[PyTree, InputT_co], OutputT_co]
LossFn = Callable[[OutputT_co, LabelT_co], jax.Array]


@flax.struct.dataclass
class TrainStepOutput:
    """Class for keeping track of an item in inventory."""

    metric: Optional[clu.metrics.Collection] = None


@flax.struct.dataclass
class EvalStepOutput:
    """Class for keeping track of an item in inventory."""

    metric: clu.metrics.Collection


ValidationStep = Callable[[PyTree, ModelT[InputT_co, OutputT_co], Batch], EvalStepOutput]


class TrainerSteps(Generic[InputT_co, OutputT_co], equinox.Module):
    @abc.abstractmethod
    def train(
        self,
        params: PyTree,
        model: ModelT[InputT_co, OutputT_co],
        batch: Batch[InputT_co, LabelT_co],
    ) -> tuple[jax.Array, TrainStepOutput]:
        """Train step"""

    validation_step = Optional[ValidationStep]


class SimpleTrainerSteps(TrainerSteps[InputT_co, OutputT_co]):
    _loss_fn: LossFn[OutputT_co, LabelT_co]
    _metrics: Type[clu.metrics.Collection]

    def __init__(
        self, loss_fn: LossFn[OutputT_co, LabelT_co], metrics: Type[clu.metrics.Collection]
    ):
        self._loss_fn = loss_fn
        self._metrics = metrics

    @jt.jaxtyped(typechecker=beartype.beartype)
    def train(
        self,
        params: PyTree,
        model: ModelT[InputT_co, OutputT_co],
        batch: Batch[InputT_co, LabelT_co],
    ) -> tuple[jax.Array, TrainStepOutput]:
        """Train for a single step."""
        inputs, labels = batch
        if labels is None:
            labels = inputs

        predictions = model(params, inputs)
        loss = self._loss_fn(predictions, labels)

        metrics = self._metrics.single_from_model_output(
            loss=jnp.astype(loss, jnp.float32),
            values=predictions,
            targets=labels,
        )

        return loss, TrainStepOutput(metrics)

    @jt.jaxtyped(typechecker=beartype.beartype)
    def validation_step(
        self,
        params: PyTree,
        model: ModelT[InputT_co, OutputT_co],
        batch: Batch,
    ) -> EvalStepOutput:
        """Evaluate for a single step."""
        inputs, labels = batch
        if labels is None:
            labels = inputs

        outputs = model(params, inputs)
        loss = self._loss_fn(outputs, labels)
        metrics = self._metrics.single_from_model_output(
            loss=jnp.astype(loss, jnp.float32), values=outputs, targets=labels
        )
        return EvalStepOutput(metrics)
