# -*- coding: utf-8 -*-
from typing import Any, Callable, Generator, Iterable
import uuid


class EventGenerator:

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


class Engine:
    """Training engine"""

    def __init__(self, model: Callable):
        self._model = model
        self._events = EventGenerator()
        self.metrics = {}

    def add_engine_listener(self, listener: EngineListener):
        return self._events.add_listener(listener)

    def remove_engine_listener(self, handle):
        return self._events.remove_listener(handle)

    def step(self, data: Iterable) -> Generator:
        """Perform one epoch"""
        self._events.fire_event(EngineListener.epoch_starting, self)
        device = get_device(self._model)

        total_batch_size = 0
        for batch_idx, batch in enumerate(data):
            x, y = _to(batch[0], device=device), _to(batch[1], device=device)
            self._events.fire_event(EngineListener.batch_starting, self, batch_idx, x, y)

            y_pred = self._model(x)
            yield EngineStep(batch_idx, x, y, y_pred)

            self._events.fire_event(EngineListener.batch_ended, self, batch_idx, x, y, y_pred)
            total_batch_size += len(x)

        self._events.fire_event(EngineListener.epoch_ended, self, total_batch_size)

    def run(self, data: Iterable, return_outputs=False) -> Optional[Any]:
        if return_outputs:
            return torch.concat(list(step.y_pred for step in self.step(data)))

        collections.deque(self.step(data), maxlen=0)
