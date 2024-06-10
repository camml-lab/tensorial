# -*- coding: utf-8 -*-
from typing import Any, Callable, ClassVar, Optional, Type, TypeVar, Union

import clu.internal.utils
import clu.metrics
import flax.struct
import jax
import jax.numpy as jnp

from . import nn_utils

Self = TypeVar("Self", bound="MetricWithCount")


@flax.struct.dataclass
class MetricWithCount(clu.metrics.Metric):
    """
    Helper class to group common functionality for metrics that can keep track using a total and
    count accumulators
    """

    total: jax.Array
    count: jax.Array
    fn: ClassVar[Callable[[jax.typing.ArrayLike], jax.Array]]

    @classmethod
    def empty(cls: Type[Self]) -> Self:
        return cls(total=jnp.array(0.0), count=jnp.array(0))

    def merge(self: Self, other: Self) -> Self:
        return type(self)(
            total=self.total + other.total,
            count=self.count + other.count,
        )

    def compute(self) -> jax.Array:
        return self.total / self.count

    @classmethod
    def from_model_output(  # pylint: disable=arguments-differ
        cls: Self,
        predictions: jax.Array,
        targets: jax.Array,
        mask: Optional[jax.Array] = None,
        **_,
    ) -> Self:
        if predictions.ndim == 0:
            predictions = predictions[None]
        if targets.ndim == 0:
            targets = targets[None]
        if mask is None:
            mask = jnp.ones_like(predictions, dtype=bool)
        else:
            mask = nn_utils.prepare_mask(mask, predictions)

        # Leading dimensions of mask and predictions must match.
        if mask.shape[0] != predictions.shape[0]:
            raise ValueError(
                f"Argument `mask` must have the same leading dimension as `values`. "
                f"Received mask of dimension {mask.shape} "
                f"and values of dimension {predictions.shape}."
            )

        mask = nn_utils.prepare_mask(mask, predictions)
        mask = mask.astype(bool)

        clu.internal.utils.check_param(mask, dtype=bool, ndim=predictions.ndim)

        predictions = jnp.where(mask, predictions, jnp.zeros_like(predictions))
        targets = jnp.where(mask, targets, jnp.zeros_like(targets))

        return cls(
            total=cls.fn(predictions - targets).sum(),
            count=jnp.where(
                mask,
                jnp.ones_like(predictions, dtype=jnp.int32),
                jnp.zeros_like(predictions, dtype=jnp.int32),
            ).sum(),
        )


@flax.struct.dataclass
class MeanSquaredError(MetricWithCount):
    total: jax.Array
    count: jax.Array
    fn = jnp.square


@flax.struct.dataclass
class MeanAbsoluteError(MetricWithCount):
    total: jax.Array
    count: jax.Array
    fn = jnp.abs


@flax.struct.dataclass
class RootMeanSquareError(clu.metrics.Metric):
    mse: MeanSquaredError

    @classmethod
    def empty(cls) -> "RootMeanSquaredError":
        return cls(mse=MeanSquaredError.empty())

    @classmethod
    def from_model_output(  # pylint: disable=arguments-differ
        cls,
        predictions: jax.Array,
        targets: jax.Array,
        mask: Optional[jax.Array] = None,
        **_,
    ) -> "MeanSquaredError":
        return cls(mse=MeanSquaredError.from_model_output(predictions, targets, mask=mask))

    def merge(self, other: "RootMeanSquareError") -> "RootMeanSquareError":
        return type(self)(mse=self.mse.merge(other.mse))

    def compute(self) -> jax.Array:
        return jnp.sqrt(self.mse.compute())


@flax.struct.dataclass
class Std(clu.metrics.Metric):
    """
    Custom version of ``clu.metrics.Std` which allows for more than just one dimensional arrays
    """

    total: jax.Array
    sum_of_squares: jax.Array
    count: jax.Array

    @classmethod
    def empty(cls):
        return cls(
            total=jnp.array(0, jnp.float32),
            sum_of_squares=jnp.array(0, jnp.float32),
            count=jnp.array(0, jnp.int32),
        )

    @classmethod
    def from_model_output(
        # pylint: disable=arguments-differ
        cls,
        values: jax.Array,
        mask: Optional[jax.Array] = None,
        **_,
    ) -> "Std":
        if values.ndim == 0:
            values = values[None]
        if mask is None:
            mask = jnp.ones(values.shape[0], dtype=jnp.int32)

        mask = nn_utils.prepare_mask(mask, values)
        return cls(
            total=values.sum(),
            sum_of_squares=jnp.where(mask, values**2, jnp.zeros_like(values)).sum(),
            count=mask.sum(),
        )

    def merge(self, other: "Std") -> "Std":
        return type(self)(
            total=self.total + other.total,
            sum_of_squares=self.sum_of_squares + other.sum_of_squares,
            count=self.count + other.count,
        )

    def compute(self) -> Any:
        # var(X) = 1/N \sum_i (x_i - mean)^2
        #        = 1/N \sum_i (x_i^2 - 2 x_i mean + mean^2)
        #        = 1/N ( \sum_i x_i^2 - 2 mean \sum_i x_i + N * mean^2 )
        #        = 1/N ( \sum_i x_i^2 - 2 mean N mean + N * mean^2 )
        #        = 1/N ( \sum_i x_i^2 - N * mean^2 )
        #        = \sum_i x_i^2 / N - mean^2
        mean = self.total / self.count
        variance = self.sum_of_squares / self.count - mean**2
        # Mathematically variance can never be negative but in reality we may run
        # into such issues due to numeric reasons.
        variance = jnp.clip(variance, min=0.0)
        return variance**0.5


# Helpers to make it easy to choose a metric using a string
metrics = {
    "mse": MeanSquaredError,
    "mae": MeanAbsoluteError,
    "rmse": RootMeanSquareError,
    "std": Std,
}


def metric(metric_type: Union[str, Type[clu.metrics.Metric]]) -> Type[clu.metrics.Metric]:
    if isinstance(metric_type, str):
        try:
            return metrics[metric_type]
        except KeyError:
            raise ValueError(f"Unknown metric {metric_type}") from None
    try:
        if issubclass(metric_type, clu.metrics.Metric):
            return metric_type
    except TypeError:
        # This happens if metric_type isn't a type
        pass

    raise TypeError(
        f"metric_type has to be a string or Metric type, got {type(metric_type).__name__}"
    )
