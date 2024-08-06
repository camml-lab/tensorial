import abc
import collections
import dataclasses
import functools
import itertools
from typing import Any, Literal, Optional

from tensorial import training

from . import optimizers as optimizers_
from . import results, utils


@dataclasses.dataclass
class MetricResult:
    meta: results.Metadata
    value: Any


MetricResults = dict[str, MetricResult]


class StageListener:
    def on_stage_starting(self, stage: "Stage"):
        """The stage is about to start"""

    def on_stage_step_start(self, stage: "Stage", step: int):
        """The stage is about to start processing a batch"""

    def on_stage_step_end(self, stage: "Stage", step: int, res: MetricResults):
        """The stage just finished processing a batch"""

    def on_stage_ending(self, stage: "Stage"):
        """Called when the stage is about to finish"""


class PassthroughStageListener(StageListener):
    """
    Pass all events that we get through to listeners subscribed to the events generator given
    to us
    """

    def __init__(self, events_generator: training.EventGenerator[StageListener]):
        self._events = events_generator

    def on_stage_starting(self, stage: "Stage"):
        self._events.fire_event(StageListener.on_stage_starting, stage)

    def on_stage_step_start(self, stage: "Stage", step: int):
        self._events.fire_event(StageListener.on_stage_step_start, stage, step)

    def on_stage_step_end(self, stage: "Stage", step: int, res: MetricResults):
        self._events.fire_event(StageListener.on_stage_step_end, stage, step, res)

    def on_stage_ending(self, stage: "Stage"):
        self._events.fire_event(StageListener.on_stage_ending, stage)


class Stage(metaclass=abc.ABCMeta):
    """Interface for loops"""

    def __init__(
        self,
        name: str,
        module,
        min_steps: Optional[int] = None,
        max_steps: int = -1,
        parent: "Stage" = None,
    ):
        self._name = name
        self._module = module
        self._min_steps = min_steps
        self._max_steps = max_steps
        self._parent = parent

        self._iteration = -1
        self.events = training.EventGenerator[StageListener]()
        self._should_stop = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def iteration(self) -> int:
        return self._iteration

    @property
    def parent(self) -> Optional["Stage"]:
        """Optional parent to the stage"""
        return self._parent

    @property
    def should_stop(self) -> bool:
        return self._should_stop

    @should_stop.setter
    def should_stop(self, stop: bool):
        self._should_stop = stop

    def run(self) -> Any:
        """Run the loop until the end or max_steps"""
        iterator = itertools.count() if self._max_steps == -1 else iter(range(self._max_steps))
        self._on_starting()

        while True:
            self._iteration = next(iterator)

            try:
                self.advance()
            except StopIteration:
                break

        self._on_stopping()

    @abc.abstractmethod
    def log(
        self,
        name: str,
        value,
        batch_size: Optional[int] = None,
        prog_bar: bool = False,
        on_step=True,
        on_epoch=True,
    ) -> None:
        """Log a result while the stage is running"""

    @abc.abstractmethod
    def _advance(self) -> Any:
        """The advance logic that should be implemented by subclasses"""

    def _on_starting(self):
        """Stage is starting"""
        self._iteration = -1
        self.events.fire_event(StageListener.on_stage_starting, self)

    def _on_advance_starting(self):
        self.events.fire_event(StageListener.on_stage_step_start, self, self._iteration)

    def advance(self) -> Any:
        """Advance the loop by one iteration"""
        assert self._max_steps == -1 or self._iteration <= self._max_steps

        if self._should_stop:
            self._should_stop = False
            raise StopIteration
        if self._max_steps != -1 and self._iteration >= self._max_steps:
            raise StopIteration

        self._on_advance_starting()
        result = self._advance()
        self._on_advance_finished(result)

    def _on_advance_finished(self, result: Any):
        self.events.fire_event(StageListener.on_stage_step_end, self, self._iteration, result)

    def _on_stopping(self):
        """The stage is stopping"""
        self.events.fire_event(StageListener.on_stage_ending, self)


class EpochStage(Stage):
    def __init__(
        self,
        name: str,
        module,
        dataloader,
        min_steps: Optional[int] = None,
        max_steps: int = -1,
        parent: "Stage" = None,
    ):
        super().__init__(name, module, min_steps=min_steps, max_steps=max_steps, parent=parent)
        self._dataloader = dataloader
        self._iterator = None
        self._batch = None
        self._metrics: Optional[results.ResultCollection] = None
        self._results: Optional[MetricResults] = None

    @property
    def metrics(self) -> results.ResultCollection:
        return self._metrics

    @property
    def batch(self):
        """Get the current batch"""
        return self._batch

    @property
    def max_batches(self) -> Optional[int]:
        return utils.sized_len(self._dataloader)

    @property
    def results(self) -> Optional[MetricResults]:
        """The results for this epoch (if any)"""
        return self._results

    def log(
        self,
        name: str,
        value,
        batch_size: Optional[int] = None,
        prog_bar: bool = False,
        on_step=True,
        on_epoch=True,
    ) -> None:
        assert self._batch is not None
        if batch_size is None:
            batch_size = utils.extract_batch_size(self._batch)

        self._metrics.log(
            self._name,
            name,
            value,
            prog_bar=prog_bar,
            batch_size=batch_size,
            batch_idx=self._iteration,
            on_step=on_step,
            on_epoch=on_epoch,
        )

    def _on_starting(self):
        self._metrics = results.ResultCollection()
        self._iterator = iter(self._dataloader)
        self._results = None
        super()._on_starting()

    def _on_advance_starting(self):
        self._batch = next(self._iterator)
        super()._on_advance_starting()

    def _on_stopping(self):
        """
        Look through the logged metrics and extract those where a user logged a results during this
        step and used the `on_step=True` option
        """
        res = {}
        for name, entry in self._metrics.items():
            if entry.meta.on_epoch:
                # Ask the metric to calculate the overall result and store it
                res[name] = MetricResult(entry.meta, entry.metric.compute())

        self._results = res
        super()._on_stopping()

    def _get_step_results(self) -> MetricResults:
        """
        Look through the logged metrics and extract those where a user logged a results during this
        step and used the `on_step=True` option
        """
        res = {}
        for name, entry in self._metrics.items():
            if entry.meta.on_step and entry.meta.batch_idx == self.iteration:
                # Get the last value (i.e. the one from this step
                res[name] = MetricResult(entry.meta, entry.last_value)

        return res

    @abc.abstractmethod
    def _advance(self) -> Any:
        """The advance logic that should be implemented by subclasses"""


class Train(EpochStage):
    def __init__(self, module, dataloader, optimizers: list[optimizers_.Optimizer], parent=None):
        super().__init__("training", module, dataloader, parent=parent)
        self._optimizers = optimizers

    def run(self) -> list[optimizers_.Optimizer]:
        super().run()
        return self._optimizers

    def _advance(self) -> MetricResults:
        res = self._module.training_step(self.batch, self._iteration)
        if self._module.automatic_optimization:
            loss, grads = res
            opt = self._optimizers[0]
            params, opt = opt.update(self._module.parameters(), grads)
            self._optimizers = [opt]
            self._module.set_parameters(params)

        return self._get_step_results()


class Validate(EpochStage):
    def __init__(self, module, dataloader, parent: Stage = None):
        super().__init__("validation", module, dataloader, parent=parent)

    def _advance(self) -> MetricResults:
        self._module.validation_step(self.batch, self._iteration)
        return self._get_step_results()


@dataclasses.dataclass
class StageInfo:
    stage: "EpochStage"
    run_every_n: int = 1


class MultiStage(Stage):
    def __init__(
        self,
        name: str,
        module,
        children: list[EpochStage],
        min_steps: Optional[int] = None,
        max_steps: int = -1,
        passthrough_listeners=True,
    ):
        super().__init__(
            name=name,
            module=module,
            min_steps=min_steps,
            max_steps=max_steps,
        )

        children = [
            StageInfo(child) if not isinstance(child, StageInfo) else child for child in children
        ]

        self._children: list[StageInfo] = children
        if passthrough_listeners:
            for info in children:
                info.stage.events.add_listener(PassthroughStageListener(self.events))

        self._running: Optional[EpochStage] = None

    def _advance(self) -> MetricResults:
        res = {}
        for info in self._children:
            if info.run_every_n % self.iteration != 0:
                continue

            child = info.stage
            self._running = child
            try:
                child.run()
            finally:
                self._running = None

            for name, entry in child.metrics.items():
                if entry.meta.on_epoch and entry.meta.batch_idx == self.iteration:
                    # Compute the final metric
                    res[name] = MetricResult(entry.meta, entry.metric.compute())

        return res

    def log(
        self,
        name: str,
        value,
        batch_size: Optional[int] = None,
        prog_bar: bool = False,
        on_step=True,
        on_epoch=True,
    ) -> None:
        """Pass logging onto the currently running stage"""
        self._running.log(name, value, batch_size, prog_bar, on_step=on_step, on_epoch=on_epoch)


class Fit(MultiStage):
    def __init__(
        self,
        module,
        train_dataloaders,
        val_dataloaders,
        optimizers: list[optimizers_.Optimizer],
        min_steps: Optional[int] = None,
        max_steps: int = -1,
        check_val_every_n_epoch: int = 1,
    ):
        children = [
            Train(module=module, dataloader=train_dataloaders, optimizers=optimizers, parent=self),
            StageInfo(
                Validate(module=module, dataloader=val_dataloaders, parent=self),
                run_every_n=check_val_every_n_epoch,
            ),
        ]

        super().__init__(
            "fit",
            module,
            children,
            min_steps=min_steps,
            max_steps=max_steps,
        )
