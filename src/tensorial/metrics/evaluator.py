from collections.abc import Callable
from typing import Any, TypeVar, Union

import beartype
import jax
import jaxtyping as jt
import reax

from tensorial import typing

from .metric import Metric

__all__ = ("Evaluator",)

OutT = TypeVar("OutT")


M = TypeVar("M", bound=Union[Metric, reax.metrics.MetricCollection])
T_co = TypeVar("T_co", covariant=True)


def _default_eval(metric: M, data: Union[tuple, Any], kwargs: dict[str, Any]) -> M:
    if not isinstance(data, tuple):
        data = (data,)
    return metric.update(*data, **kwargs)


class Evaluator:
    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        metric: Union[
            type[reax.metrics.Metric], reax.metrics.Metric, reax.metrics.MetricCollection
        ],
        eval_fn: Callable[[M, T_co, ...], M] = None,
    ):
        self._metric = metric
        self._eval_fn = eval_fn or _default_eval

    @jt.jaxtyped(typechecker=beartype.beartype)
    def evaluate(
        self, loader: typing.DataLoader[T_co], **kwargs
    ) -> Union[jax.Array, dict[str, jax.Array]]:
        updater = self._metric.empty()
        for data in loader:
            updater = self._eval_fn(updater, data, kwargs)

        return updater.compute()
