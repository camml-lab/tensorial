import sys
from typing import Optional

import tqdm

from tensorial.jaxning import stages, training


class TqdmProgressBar(training.TrainerListener):
    BAR_FORMAT = (
        "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}{postfix}]"
    )

    def __init__(self):
        self._bar: Optional[tqdm.tqdm] = None

    def on_epoch_starting(self, trainer: "training.Trainer", stage: stages.EpochStage) -> None:
        self._bar = tqdm.tqdm(
            total=stage.max_batches,
            desc=stage.name,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
            bar_format=self.BAR_FORMAT,
        )

    def on_batch_ending(
        self,
        trainer: "training.Trainer",
        stage: stages.EpochStage,
        batch_idx: int,
        metrics: stages.MetricResults,
    ) -> None:
        self._bar.n = batch_idx + 1
        postfix = {}
        if metrics:
            for name, result in metrics.items():
                if result.meta.prog_bar:
                    postfix[result.meta.name] = result.value
        if postfix:
            self._bar.set_postfix(postfix)

        self._bar.refresh()

    def on_epoch_ending(
        self, trainer: "training.Trainer", stage: stages.EpochStage, metrics: dict
    ) -> None:
        pass

    def on_stage_ending(self, trainer: "training.Trainer", stage: stages.Stage) -> None:
        self._bar.close()
