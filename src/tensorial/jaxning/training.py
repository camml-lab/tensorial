import contextlib
import signal
from typing import Literal, Optional

import jax

from tensorial import training

from . import stages
from .optimizers import Optimizer

__all__ = ("Trainer",)


class TrainerListener:
    def on_stage_starting(self, trainer: "Trainer", stage: stages.Stage) -> None:
        """A trainer stage is about to begin"""

    def on_epoch_starting(self, trainer: "Trainer", stage: stages.EpochStage) -> None:
        """An epoch is just about to begin"""

    def on_batch_starting(
        self, trainer: "Trainer", stage: stages.EpochStage, batch_idx: int
    ) -> None:
        """A batch is just about to be processed"""

    def on_batch_ending(
        self, trainer: "Trainer", stage: stages.EpochStage, batch_idx: int, metrics: dict
    ) -> None:
        """A batch has just been processed"""

    def on_epoch_ending(self, trainer: "Trainer", stage: stages.EpochStage, metrics: dict) -> None:
        """An epoch is ending"""

    def on_stage_ending(self, trainer: "Trainer", stage: stages.Stage) -> None:
        """A trainer stage is ending"""


class Trainer(stages.StageListener):
    def __init__(
        self,
        accelerator: Literal["auto", "cpu", "gpu"] = "auto",
        log_every_n_steps: int = 50,
        check_val_every_n_epoch: int = 1,
    ):
        self._accelerator = (
            jax.devices()[0] if accelerator == "auto" else jax.devices(accelerator)[0]
        )
        self._log_every_n_steps = log_every_n_steps
        self.check_val_every_n_epoch = check_val_every_n_epoch

        self._automatic_optimization = True
        self._optimizers = []
        self._stage: Optional[stages.Stage] = None
        self._current_epoch: Optional[int] = None

        self.events = training.EventGenerator[TrainerListener]()

        self._num_batches = None

    @property
    def current_epoch(self) -> Optional[int]:
        return self._current_epoch

    @property
    def optimizers(self) -> list[Optimizer]:
        return self._optimizers

    @optimizers.setter
    def optimizers(self, opts: list[Optimizer]) -> None:
        self._optimizers = opts

    @property
    def should_stop(self):
        return self._stage.should_stop

    @should_stop.setter
    def should_stop(self, stop: bool):
        self._stage.should_stop = stop

    def log(
        self,
        name: str,
        value,
        prog_bar: bool = False,
        batch_size: Optional[int] = None,
        logger: bool = None,
        on_step=True,
        on_epoch=True,
    ) -> None:
        if self._stage is None:
            raise RuntimeError(
                "Logging is only supported during one of the train/validate/test stages. "
                "There is currently not stage running."
            )

        self._stage.log(
            name, value, prog_bar=prog_bar, batch_size=batch_size, on_step=on_step, on_epoch=on_step
        )

    def fit(
        self,
        module,
        train_dataloaders=None,
        val_dataloaders=None,
        datamodule=None,
        ckpt_path=None,
        max_epochs: int = 1_000,
        min_epochs: Optional[int] = None,
    ):
        module.trainer = self
        try:
            batch = next(iter(train_dataloaders))
            module.configure_model(batch)
            self._configure_optimizers(module)

            fit = stages.Fit(
                module,
                train_dataloaders,
                val_dataloaders,
                optimizers=self.optimizers,
                min_steps=min_epochs,
                max_steps=max_epochs,
                check_val_every_n_epoch=self.check_val_every_n_epoch,
            )
            self._run_stage(fit)
            self._current_epoch = None
        finally:
            module.trainer = None

    def _run_stage(self, stage: stages.Stage) -> stages.Stage:
        try:
            with self._attach(stage):
                with stage.events.listen_context(self):
                    stage.run()
        except KeyboardInterrupt:
            # Disable further Ctrl+C presses while we respond to this one
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            exit(1)

        return stage

    @contextlib.contextmanager
    def _attach(self, stage: stages.Stage):
        self._stage = stage
        try:
            yield
        finally:
            self._stage = None

    def _configure_optimizers(self, module):
        opts = module.configure_optimizers()
        if not isinstance(opts, list):
            opts = [opts]
        self.optimizers = list(map(lambda opt: Optimizer(*opt), opts))

    def on_stage_starting(self, stage: "stages.Stage") -> None:
        """The stage is about to start"""
        self.events.fire_event(TrainerListener.on_stage_starting, self, stage)

        if isinstance(stage, stages.EpochStage):
            self.events.fire_event(TrainerListener.on_epoch_starting, self, stage)

    def on_stage_step_start(self, stage: "stages.Stage", step: int):
        if isinstance(stage, stages.EpochStage):
            self.events.fire_event(TrainerListener.on_batch_starting, self, stage, step)

    def on_stage_step_end(self, stage: "stages.Stage", step: int, metrics: dict):
        if isinstance(stage, stages.EpochStage):
            self.events.fire_event(
                TrainerListener.on_batch_ending,
                self,
                stage,
                batch_idx=step,
                metrics=metrics,
            )

    def on_stage_ending(self, stage: "stages.Stage") -> None:
        """Called when the stage has finished a full epoch"""
        if isinstance(stage, stages.EpochStage):
            self.events.fire_event(TrainerListener.on_epoch_ending, self, stage, stage.results)

        self.events.fire_event(TrainerListener.on_stage_ending, self, stage)
