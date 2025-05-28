from collections.abc import Iterable
from typing import Generic, TypeVar

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

    def items(self) -> Iterable[tuple[str, T]]:
        return self._registry.items()

    def register(self, name: str, obj: T):
        self._registry[name] = obj

    def register_many(self, objects: dict[str, T]):
        for vals in objects.items():
            self.register(*vals)

    def unregister(self, name: str) -> T:
        return self._registry.pop(name)

    def find(self, starts_with: str) -> Iterable[tuple[str, T]]:
        for name, obj in self._registry.items():
            if name.startswith(starts_with):
                yield name, obj
