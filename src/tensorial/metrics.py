# -*- coding: utf-8 -*-
from typing import Optional, Type, Union

import clu.internal.utils
import clu.metrics
import flax.struct
import jax
import jax.numpy as jnp


@flax.struct.dataclass
class MeanSquaredError(clu.metrics.Metric):
    total: jnp.array
    count: jnp.array

    @classmethod
    def empty(cls) -> 'MeanSquaredError':
        return cls(total=jnp.array(0, jnp.float32), count=jnp.array(0, jnp.int32))

    @classmethod
    def from_model_output(  # pylint: disable=arguments-differ
            cls, predictions: jax.Array, targets: jax.Array, mask: Optional[jnp.array] = None, **_
    ) -> 'MeanSquaredError':
        if predictions.ndim == 0:
            predictions = predictions[None]
        if targets.ndim == 0:
            targets = targets[None]
        if mask is None:
            mask = jnp.ones_like(predictions)

        # Leading dimensions of mask and predictions must match.
        if mask.shape[0] != predictions.shape[0]:
            raise ValueError(
                f'Argument `mask` must have the same leading dimension as `values`. '
                f'Received mask of dimension {mask.shape} '
                f'and values of dimension {predictions.shape}.'
            )

        # Broadcast mask to the same number of dimensions as values.
        if mask.ndim < predictions.ndim:
            mask = jnp.expand_dims(mask, axis=tuple(jnp.arange(mask.ndim, predictions.ndim)))
        mask = mask.astype(bool)

        clu.internal.utils.check_param(mask, dtype=bool, ndim=predictions.ndim)

        predictions = jnp.where(mask, predictions, jnp.zeros_like(predictions))
        targets = jnp.where(mask, targets, jnp.zeros_like(targets))

        return cls(
            total=jnp.square(predictions - targets).sum(),
            count=jnp.where(
                mask, jnp.ones_like(predictions, dtype=jnp.int32), jnp.zeros_like(predictions, dtype=jnp.int32)
            ).sum(),
        )

    def merge(self, other: 'MeanSquaredError') -> 'MeanSquaredError':
        return type(self)(
            total=self.total + other.total,
            count=self.count + other.count,
        )

    def compute(self) -> jax.Array:
        return self.total / self.count


@flax.struct.dataclass
class MeanAbsoluteError(clu.metrics.Metric):
    total: jnp.array
    count: jnp.array

    @classmethod
    def empty(cls) -> 'MeanAbsoluteError':
        return cls(total=jnp.array(0, jnp.float32), count=jnp.array(0, jnp.int32))

    @classmethod
    def from_model_output(  # pylint: disable=arguments-differ
            cls, predictions: jax.Array, targets: jax.Array, mask: Optional[jnp.array] = None, **_
    ) -> 'MeanAbsoluteError':
        if predictions.ndim == 0:
            predictions = predictions[None]
        if targets.ndim == 0:
            targets = targets[None]
        if mask is None:
            mask = jnp.ones_like(predictions)

        # Leading dimensions of mask and predictions must match.
        if mask.shape[0] != predictions.shape[0]:
            raise ValueError(
                f'Argument `mask` must have the same leading dimension as `values`. '
                f'Received mask of dimension {mask.shape} '
                f'and values of dimension {predictions.shape}.'
            )

        # Broadcast mask to the same number of dimensions as values.
        if mask.ndim < predictions.ndim:
            mask = jnp.expand_dims(mask, axis=tuple(jnp.arange(mask.ndim, predictions.ndim)))
        mask = mask.astype(bool)

        clu.internal.utils.check_param(mask, dtype=bool, ndim=predictions.ndim)

        predictions = jnp.where(mask, predictions, jnp.zeros_like(predictions))
        targets = jnp.where(mask, targets, jnp.zeros_like(targets))

        return cls(
            total=jnp.abs(predictions - targets).sum(),
            count=jnp.where(
                mask, jnp.ones_like(predictions, dtype=jnp.int32), jnp.zeros_like(predictions, dtype=jnp.int32)
            ).sum(),
        )

    def merge(self, other: 'MeanSquaredError') -> 'MeanSquaredError':
        return type(self)(
            total=self.total + other.total,
            count=self.count + other.count,
        )

    def compute(self) -> jax.Array:
        return self.total / self.count


@flax.struct.dataclass
class RootMeanSquareError(clu.metrics.Metric):
    mse: MeanSquaredError

    @classmethod
    def empty(cls) -> 'RootMeanSquaredError':
        return cls(mse=MeanSquaredError.empty())

    @classmethod
    def from_model_output(  # pylint: disable=arguments-differ
            cls, predictions: jnp.array, targets: jax.Array, mask: Optional[jnp.array] = None, **_
    ) -> 'MeanSquaredError':
        return cls(mse=MeanSquaredError.from_model_output(predictions, targets, mask=mask))

    def merge(self, other: 'RootMeanSquareError') -> 'RootMeanSquareError':
        return type(self)(mse=self.mse.merge(other.mse))

    def compute(self) -> jax.Array:
        return jnp.sqrt(self.mse.compute())


# Helpers to make it easy to choose a metric using a string
metrics = {
    'mse': MeanSquaredError,
    'mae': MeanAbsoluteError,
    'rmse': RootMeanSquareError,
}


def metric(metric_type: Union[str, Type[clu.metrics.Metric]]) -> Type[clu.metrics.Metric]:
    if isinstance(metric_type, str):
        try:
            return metrics[metric_type]
        except KeyError:
            raise ValueError(f'Unknown metric {metric_type}') from None
    try:
        if issubclass(metric_type, clu.metrics.Metric):
            return metric_type
    except TypeError:
        # This happens if metric_type isn't a type
        pass

    raise TypeError(f'metric_type has to be a string or Metric type, got {type(metric_type).__name__}')
