import dataclasses
from typing import Any, Optional, Union

import jax

from tensorial import metrics

_VALUE = Union[jax.Array, metrics.Metric]


@dataclasses.dataclass
class Metadata:
    fx: str
    name: str
    batch_idx: int
    prog_bar: bool
    on_step: bool
    on_epoch: bool
    last_value: Optional[metrics.Metric] = None


# @dataclasses.dataclass
class ArrayResultMetric(metrics.Metric[jax.Array]):
    value: jax.Array  # Redefine here so the typing hinting works
    cumulated_batch_size: int

    @classmethod
    def create(cls, value: _VALUE, batch_size: int) -> "ArrayResultMetric":
        return ArrayResultMetric(value=value, cumulated_batch_size=batch_size)

    def merge(self, other: "ArrayResultMetric") -> "ArrayResultMetric":
        return ArrayResultMetric(
            value=self.value + other.value,
            cumulated_batch_size=self.cumulated_batch_size + self.cumulated_batch_size,
        )

    def compute(self) -> jax.Array:
        return self.value / self.cumulated_batch_size


class ResultEntry:
    def __init__(self, meta: Metadata, metric: metrics.Metric, last_value=None):
        self._meta = meta  # Readonly
        self.metric = metric
        self.last_value = last_value

    @property
    def meta(self) -> Metadata:
        return self._meta


class ResultCollection(dict[str, ResultEntry]):
    """A dictionary holding model metrics"""

    def __str__(self) -> str:
        my_str = str({key: val for key, val in self.items()})
        return f"{type(self)}.__name__({my_str})"

    def log(
        self,
        fx: str,
        name: str,
        value: _VALUE,
        batch_idx: int,
        prog_bar: bool = False,
        on_step: bool = False,
        on_epoch: bool = True,
        batch_size: Optional[int] = None,
    ):
        key = f"{fx}.{name}"

        if isinstance(value, jax.typing.ArrayLike):
            metric = ArrayResultMetric.create(value, batch_size)
        elif isinstance(value, metrics.Metric):
            metric = value
        else:
            raise TypeError(
                f"Value must be a `metrics.Metric` or a raw value, got {type(value).__name__}"
            )

        meta = Metadata(
            fx=fx,
            name=name,
            batch_idx=batch_idx,
            prog_bar=prog_bar,
            on_step=on_step,
            on_epoch=on_epoch,
        )
        last_value = metric.compute() if on_step else None

        if key in self:
            metric = self[key].metric.merge(metric)

        self[key] = ResultEntry(meta, metric, last_value=last_value)
