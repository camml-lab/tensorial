from collections.abc import Sequence
from typing import Any, Union

import beartype
import equinox
import flax.core
import jaxtyping as jt
from reax.metrics import Metric

__all__ = ("MetricCollection",)


MetricType = Union[type[Metric], Metric]


class MetricCollection(equinox.Module):
    _metrics: flax.core.FrozenDict[str, Metric]

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(self, metrics: Union[MetricType, Sequence[MetricType], dict[str, MetricType]]):
        super().__init__()
        if isinstance(metrics, Metric):
            metrics = [metrics]
        if isinstance(metrics, Sequence):
            metrics = {type(metric).__name__: metric for metric in metrics}

        if not isinstance(metrics, dict):
            raise ValueError(
                f"Must pass in a metric, sequence of metrics or a dictionary of metrics, "
                f"got {type(metrics).__name__}"
            )

        metric_instances = {}
        for name, metric in metrics.items():
            if isinstance(metric, Metric):
                metric_instances[name] = metric
            elif issubclass(metric, Metric):
                metric_instances[name] = metric()
            else:
                raise TypeError(f"Expected a metric class or instance, got {type(metric).__name__}")

        self._metrics = flax.core.FrozenDict(metrics)

    def empty(self) -> "MetricCollection":
        """Create a new empty instance.

        By default, this will call the constructor with no arguments, if needed, subclasses can
        overwrite this with custom behaviour.
        """
        return MetricCollection({name: metric.empty() for name, metric in self._metrics.items()})

    def create(self, *args, **kwargs) -> "MetricCollection":
        return MetricCollection(
            {name: metric.create(*args, **kwargs) for name, metric in self._metrics.items()}
        )

    def update(self, *args, **kwargs) -> "MetricCollection":
        return MetricCollection(
            {name: metric.update(*args, **kwargs) for name, metric in self._metrics.items()}
        )

    @jt.jaxtyped(typechecker=beartype.beartype)
    def merge(self, other: "MetricCollection") -> "MetricCollection":
        merged = {}
        for name, metric in self._metrics.items():
            try:
                merged[name] = metric.merge(
                    other._metrics[name]  # pylint: disable=protected-access
                )
            except KeyError:
                merged[name] = metric  # Nothing to merge

        # Now put in all the ones that only appear in 'other'
        for name in set(other._metrics.keys()) - set(  # pylint: disable=protected-access
            self._metrics.keys()
        ):
            merged[name] = other._metrics[name]  # pylint: disable=protected-access

        return MetricCollection(merged)

    def compute(self) -> dict[str, Any]:
        return {name: metric.compute() for name, metric in self._metrics.items()}
