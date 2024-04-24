# -*- coding: utf-8 -*-
import contextlib
import logging
import math
from typing import Any, Callable, List
import uuid

from tensorial import training

TRAIN_OVERFITTING = 'overfitting'

__all__ = 'TRAIN_OVERFITTING', 'EventGenerator', 'TrainerListener', 'TrainingLogger', 'EarlyStopping', 'MetricsLogging'

_LOGGER = logging.getLogger(__name__)


class EventGenerator:
    """Manage listeners and fire events"""

    def __init__(self):
        self._event_listeners = {}

    def add_listener(self, listener) -> Any:
        handle = uuid.uuid4()
        self._event_listeners[handle] = listener
        return handle

    def remove_listener(self, handle) -> Any:
        return self._event_listeners.pop(handle)

    def fire_event(self, event_fn: Callable, *args, **kwargs):
        for listener in self._event_listeners.values():
            getattr(listener, event_fn.__name__)(*args, **kwargs)

    @contextlib.contextmanager
    def listen_context(self, *listener: 'TrainerListener'):
        uuids = tuple()
        try:
            uuids = tuple(map(self.add_listener, listener))
            yield
        finally:
            tuple(map(self.remove_listener, uuids))


class TrainerListener:

    def on_epoch_starting(self, trainer: 'training.Trainer', epoch_num: int):
        """A training epoch has started"""

    def on_epoch_finishing(self, trainer: 'training.Trainer', epoch_num: int):
        """An epoch is finishing but the model state has not been updated yet"""

    def on_epoch_finished(self, trainer: 'training.Trainer', epoch_num: int):
        """An epoch has finished and the model state has been updated"""

    def on_training_stopping(self, trainer: 'training.Trainer', stop_msg: str):
        """A training run is stopping"""


class TrainingLogger(TrainerListener):
    """Log metrics for training and validation"""

    def __init__(self, log_every_n_epochs: int = 1, log_on_stop: bool = True):
        self._log: List[dict] = []
        self._log_ever_n_epochs = log_every_n_epochs
        self._log_on_stop = log_on_stop

    def raw_log(self) -> List[dict]:
        return self._log

    def as_dataframe(self):
        import pandas
        return pandas.DataFrame(self._log)

    def on_epoch_finished(self, trainer: 'training.Trainer', epoch_num: int):
        if epoch_num % self._log_ever_n_epochs == 0:
            self._save_log(trainer, epoch_num)

    def _save_log(self, trainer: 'training.Trainer', epoch):
        log_entry = {'epoch': epoch}
        if trainer.train_metrics is not None:
            log_entry.update({'training_' + key: value for key, value in trainer.train_metrics.items()})
        if trainer.validate_metrics is not None:
            log_entry.update({'validation_' + key: value for key, value in trainer.validate_metrics.items()})
        self._log.append(log_entry)


class EarlyStopping(TrainerListener):
    """
    Basic early stopping class.  If, during training, the chosen metric (defaults to loss) increases by `min_delta` for
    `patience` steps in a row then the training is stopped.
    """

    _best_value = float('inf')

    def __init__(self, patience: int, metric='loss', min_delta: float = 0.):
        self._patience = patience
        self._metric = metric
        self._num_increases = 0
        self._min_delta = min_delta
        self._has_improved = False

    @property
    def patience(self) -> int:
        return self._patience

    @property
    def has_improved(self) -> bool:
        return self._has_improved

    def on_epoch_finished(self, trainer: 'training.Trainer', epoch_num):
        new_value = trainer.validate_metrics[self._metric]
        if math.isinf(self._best_value) or self._best_value - new_value > self._min_delta:
            # Reset
            self._num_increases = 0
            self._best_value = new_value
            self._has_improved = True
        else:
            self._num_increases += 1
            self._has_improved = False

        if self._num_increases > self.patience:
            trainer.stop(TRAIN_OVERFITTING)


class MetricsLogging(TrainerListener):
    """
    See https://docs.python.org/3/library/stdtypes.html#old-string-formatting for formatting style
    """

    # Only these quantities are available in the default metrics (and maybe not even validation if it is not supplied)
    DEFAULT_MSG = '%(epoch)i: %(training_loss).2f %(validation_loss).2f'

    def __init__(self, log_level=logging.INFO, msg=DEFAULT_MSG, log_every: int = 10):
        self._log_level = log_level
        self._msg = msg
        self._log_every = log_every

    def on_epoch_finished(self, trainer: 'training.Trainer', epoch_num: int):
        if epoch_num % self._log_every == 0:
            _LOGGER.log(self._log_level, self._msg, trainer.metrics_log.raw_log()[-1])
