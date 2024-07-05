import jax

from . import training


class TrainingProfiler(training.TrainerListener):
    """
    This profiler will automatically start profiling when training starts and stop when it ends.
    The user can optionally manually start and stop also"""

    def __init__(self, log_dir: str):
        self._log_dir = log_dir
        self._profiling = False

    def start(self) -> bool:
        if self._profiling:
            return False

        jax.profiler.start_trace(self._log_dir)
        self._profiling = True
        return True

    def stop(self) -> bool:
        if not self._profiling:
            return False

        jax.profiler.stop_trace()
        self._profiling = False
        return False

    def on_training_starting(self, trainer: "training.Trainer"):
        self.start()

    def on_training_stopping(self, trainer: "training.Trainer", stop_msg: str):
        self.stop()


class TrainingWriter(training.TrainerListener):
    import tensorboardX

    tbx = tensorboardX

    def __init__(self, log_dir: str):
        self._log_dir = log_dir
        self._writer = None

    def on_training_starting(self, trainer: "training.Trainer"):
        self._writer = self.tbx.SummaryWriter(self._log_dir)

    def on_epoch_finished(self, trainer: "training.Trainer", epoch_num: int):
        for name, value in trainer.train_metrics.items():
            self._writer.add_scalar(f"{name}/train", value, epoch_num)

        for name, value in trainer.validate_metrics.items():
            self._writer.add_scalar(f"{name}/validate", value, epoch_num)

        self._writer.flush()

    def on_training_stopping(self, trainer: "training.Trainer", stop_msg: str):
        self._writer.close()
        self._writer = None
