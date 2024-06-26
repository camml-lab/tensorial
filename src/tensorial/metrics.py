from collections.abc import Callable
import functools
from typing import Any, ClassVar, Optional, TypeVar

import beartype
import clu.internal.utils
import clu.metrics
import flax.struct
import jax
import jax.numpy as jnp
import jaxtyping as jt

from . import _utils, nn_utils, typing

__all__ = (
    "MetricWithCount",
    "MeanSquaredError",
    "MeanAbsoluteError",
    "RootMeanSquareError",
    "Std",
    "Min",
    "Max",
    "Unique",
)

M = TypeVar("M", bound="Metric")


@flax.struct.dataclass
class MetricWithCount(clu.metrics.Metric):
    """
    Helper class to group common functionality for metrics that can keep track using a total and
    count accumulators
    """

    Self = TypeVar("Self", bound="MetricWithCount")

    total: jax.Array
    count: jax.Array
    fn: ClassVar[Callable[[jax.typing.ArrayLike], jax.Array]]

    @classmethod
    def empty(cls: type[Self]) -> Self:
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
            values: jax.Array,
            targets: jax.Array,
            *_,
            mask: Optional[jax.Array] = None,
            **__,
    ) -> Self:
        if values.ndim == 0:
            values = values[None]
        if targets.ndim == 0:
            targets = targets[None]
        if mask is None:
            mask = jnp.ones_like(values, dtype=bool)
        else:
            mask = nn_utils.prepare_mask(mask, values)

        # Leading dimensions of mask and predictions must match.
        if mask.shape[0] != values.shape[0]:
            raise ValueError(
                f"Argument `mask` must have the same leading dimension as `values`. "
                f"Received mask of dimension {mask.shape} "
                f"and values of dimension {values.shape}."
            )

        mask = nn_utils.prepare_mask(mask, values)
        mask = mask.astype(bool)

        clu.internal.utils.check_param(mask, dtype=bool, ndim=values.ndim)

        values = jnp.where(mask, values, jnp.zeros_like(values))
        targets = jnp.where(mask, targets, jnp.zeros_like(targets))

        return cls(
            total=cls.fn(values - targets).sum(),
            count=jnp.where(
                mask,
                jnp.ones_like(values, dtype=jnp.int32),
                jnp.zeros_like(values, dtype=jnp.int32),
            ).sum(),
        )


@flax.struct.dataclass
class MeanSquaredError(MetricWithCount):
    fn = jnp.square


@flax.struct.dataclass
class MeanAbsoluteError(MetricWithCount):
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
            values: jax.Array,
            targets: jax.Array,
            *_,
            mask: Optional[jax.Array] = None,
            **__,
    ) -> "MeanSquaredError":
        return cls(mse=MeanSquaredError.from_model_output(values, targets, mask=mask))

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
            *_,
            mask: Optional[jax.Array] = None,
            **__,
    ) -> "Std":
        if values.ndim == 0:
            values = values[None]
        if mask is None:
            mask = jnp.ones(values.shape[0], dtype=jnp.int32)

        mask = nn_utils.prepare_mask(mask, values)
        return cls(
            total=values.sum(),
            sum_of_squares=jnp.where(mask, values ** 2, jnp.zeros_like(values)).sum(),
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
        variance = self.sum_of_squares / self.count - mean ** 2
        # Mathematically variance can never be negative but in reality we may run
        # into such issues due to numeric reasons.
        variance = jnp.clip(variance, min=0.0)
        return variance ** 0.5


@flax.struct.dataclass
class StatMetric(clu.metrics.Metric):
    """
    todo:
        * Rename this class to something more intuitive
    """

    value: jax.Array = None
    fn: ClassVar[Callable[[jax.typing.ArrayLike], jax.typing.ArrayLike]]
    merge_fn: ClassVar[Callable]

    @classmethod
    def empty(cls: type[M]) -> M:
        return cls(value=None)

    @classmethod
    def create(
            cls,
            fun: Callable[[jax.typing.ArrayLike], jax.typing.ArrayLike],
            merge_fun: Callable = jnp.append,
    ):  # No way to annotate return type
        @flax.struct.dataclass
        class Metric(cls):
            """Wrapper Metric class that collects output after applying `fun`."""

            fn = fun
            merge_fn = merge_fun

            @classmethod
            def from_model_output(
                    cls: type[M],
                    values: jax.typing.ArrayLike,
                    *_,
                    mask: Optional[jax.typing.ArrayLike] = None,
                    **__,
            ) -> M:
                if mask is None:
                    mask = jnp.ones_like(values, dtype=bool)
                else:
                    mask = nn_utils.prepare_mask(mask, values)

                # Leading dimensions of mask and predictions must match.
                if mask.shape[0] != values.shape[0]:
                    raise ValueError(
                        f"Argument `mask` must have the same leading dimension as `values`. "
                        f"Received mask of dimension {mask.shape} "
                        f"and values of dimension {values.shape}."
                    )

                clu.internal.utils.check_param(mask, dtype=bool, ndim=values.ndim)
                values = jnp.where(mask, values, jnp.zeros_like(values))

                return cls(value=cls.fn(values))

        return Metric

    def merge(self, other) -> "StatMetric":
        if self.value is None:
            if other.value is None:
                return self
            return other

        cls = type(self)
        return cls(value=cls.fn(cls.merge_fn(self.value, other.value)))

    def compute(self) -> jax.Array:
        return self.value


def _default_eval_fn(data: Any, metrics: type[clu.metrics.Collection]) -> clu.metrics.Collection:
    if isinstance(data, tuple):
        if len(data) == 2:
            return metrics.single_from_model_output(values=data[0], targets=data[1])

        return metrics.single_from_model_output(values=data)

    return metrics.single_from_model_output(values=data)


T_co = TypeVar("T_co", covariant=True)


class Evaluator:
    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
            self,
            metrics: type[clu.metrics.Collection],
            eval_fn: Callable[
                [T_co, type[clu.metrics.Collection], ...], clu.metrics.Collection
            ] = _default_eval_fn,
    ):
        self._metrics = metrics
        self._eval_fn = eval_fn

    @jt.jaxtyped(typechecker=beartype.beartype)
    def evaluate(self, loader: typing.DataLoader[T_co], **kwargs) -> dict[str, jax.Array]:
        updater = self._metrics.empty()
        for data in loader:
            updater = updater.merge(self._eval_fn(data, self._metrics, **kwargs))
        return updater.compute()


@jt.jaxtyped(typechecker=beartype.beartype)
def evaluate_stats(
        metrics: type[clu.metrics.Collection], loader: typing.DataLoader[T_co]
) -> dict[str, jax.Array]:
    """
    Evaluate metrics directly on a dataset, collecting stats about the data itself.
    :param metrics: the metrics to calculate
    :param loader: the data loader
    :return: a dictionary containing the evaluated metrics
    """
    return Evaluator(metrics).evaluate(loader)


class NumUnique(StatMetric):
    """Count the number of unique entries"""

    fn = jnp.unique

    def compute(self) -> jax.Array:
        return jnp.asarray(jnp.size(self.value))


Min = StatMetric.create(jnp.min)
Max = StatMetric.create(jnp.max)
Unique = StatMetric.create(jnp.unique)


class LeastSquaresEstimate(clu.metrics.Metric):
    inputs: Optional[jax.Array]
    outputs: Optional[jax.Array]

    @classmethod
    def empty(cls) -> "LeastSquaresEstimate":
        return LeastSquaresEstimate(inputs=None, outputs=None)

    @classmethod
    def from_model_output(cls, inputs: jax.Array, outputs: jax.Array) -> "LeastSquaresEstimate":
        return cls(inputs=inputs, values=outputs)

    def compute(self) -> jax.Array:
        return jnp.linalg.lstsq(self.inputs, self.outputs)[0]

    def merge(self, other: "LeastSquaresEstimate") -> "LeastSquaresEstimate":
        if not self.inputs.shape:
            return other

        return LeastSquaresEstimate(
            type_values=jnp.concatenate((self.inputs, other.inputs)),
            values=jnp.concatenate((self.outputs, other.outputs)),
        )


# Helpers to make it easy to choose a metric using a string
registry = _utils.Registry[type(clu.metrics.Metric)](
    {
        "mse": MeanSquaredError,
        "mae": MeanAbsoluteError,
        "rmse": RootMeanSquareError,
        "std": Std,
        "min": Min,
        "max": Max,
        "unique": Unique,
    }
)


@functools.singledispatch
def metric(metric_type: type[clu.metrics.Metric]) -> type[clu.metrics.Metric]:
    if issubclass(metric_type, clu.metrics.Metric):
        return metric_type

    raise TypeError(
        f"metric_type has to be a string or Metric type, got {type(metric_type).__name__}"
    )


@metric.register
def metric(metric_type: str) -> type[clu.metrics.Metric]:
    global registry
    return registry[metric_type]
