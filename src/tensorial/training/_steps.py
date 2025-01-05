import abc
from collections.abc import Callable
from typing import Any, TypeVar

import beartype
import equinox
import jax
import jaxtyping as jt

from tensorial import metrics as metrics_mod

PyTree = Any
InputT_co = TypeVar("InputT_co", covariant=True)
OutputT_co = TypeVar("OutputT_co", covariant=True)
LabelT_co = TypeVar("LabelT_co", covariant=True)
Batch = tuple[InputT_co, LabelT_co]
ModelT = Callable[[PyTree, InputT_co], OutputT_co]
LossFn = Callable[[OutputT_co, LabelT_co], jax.Array]


class StepOut(equinox.Module):
    metrics: metrics_mod.MetricCollection


class TrainerSteps(equinox.Module):
    @abc.abstractmethod
    def training_step(
        self,
        params: PyTree,
        model: ModelT[InputT_co, OutputT_co],
        batch: Batch[InputT_co, LabelT_co],
    ) -> tuple[jax.Array, StepOut]:
        """Train step"""

    @abc.abstractmethod
    def validation_step(
        self,
        params: PyTree,
        model: ModelT[InputT_co, OutputT_co],
        batch: Batch[InputT_co, LabelT_co],
    ) -> tuple[jax.Array, StepOut]:
        """Validate step"""


class SimpleTrainerSteps(TrainerSteps):
    _loss_fn: LossFn[OutputT_co, LabelT_co]
    _metrics: metrics_mod.MetricCollection

    def __init__(
        self, loss_fn: LossFn[OutputT_co, LabelT_co], metrics: metrics_mod.MetricCollection
    ):
        self._loss_fn = loss_fn
        self._metrics = metrics

    @jt.jaxtyped(typechecker=beartype.beartype)
    def training_step(
        self,
        params: PyTree,
        model: ModelT[InputT_co, OutputT_co],
        batch: Batch[InputT_co, LabelT_co],
    ) -> tuple[jax.Array, StepOut]:
        """Train for a single step."""
        inputs, labels = batch
        if labels is None:
            labels = inputs

        predictions = model(params, inputs)
        loss = self._loss_fn(predictions, labels)
        metrics = self._metrics.create(predictions, labels) if self._metrics is not None else None

        return loss, StepOut(metrics=metrics)

    @jt.jaxtyped(typechecker=beartype.beartype)
    def validation_step(
        self,
        params: PyTree,
        model: ModelT[InputT_co, OutputT_co],
        batch: Batch[InputT_co, LabelT_co],
    ) -> tuple[jax.Array, StepOut]:
        """Validate for a single step."""
        inputs, labels = batch
        if labels is None:
            labels = inputs

        predictions = model(params, inputs)
        loss = self._loss_fn(predictions, labels)
        metrics = self._metrics.create(predictions, labels)

        return loss, StepOut(metrics=metrics)
