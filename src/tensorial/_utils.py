# -*- coding: utf-8 -*-
from typing import Generic, Iterator, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Simple registry of objects with unique names"""

    def __init__(self, init: dict[str, T] = None):
        self._registry = {}
        if init:
            self.register_many(init)

    def __len__(self) -> int:
        return len(self._registry)

    def __getitem__(self, item: str) -> T:
        return self._registry[item]

    def __iter__(self):
        return iter(self._registry)

    def items(self) -> Iterator[tuple[str, T]]:
        return self._registry.items()

    def register(self, name: str, obj: T):
        self._registry[name] = obj

    def register_many(self, objects: dict[str, T]):
        [self.register(*vals) for vals in objects.items()]

    def unregister(self, name: str) -> T:
        return self._registry.pop(name)
