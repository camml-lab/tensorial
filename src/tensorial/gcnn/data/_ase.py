"""Module for loading ase.Atoms objects as graphs"""

from collections.abc import Sequence
import importlib.util
import sys
from typing import Any, Final

import jraph

from .. import atomic

__all__ = ("AseDataLoader",)


def lazy_import(name: str):
    """Lazily import a module using the standard library."""
    spec = importlib.util.find_spec(name)
    if spec is None:
        # Optimization: if it's completely missing, we can fail early
        # or return a dummy object.
        pass
    loader = importlib.util.LazyLoader(spec.loader)
    module = importlib.util.module_from_spec(spec)
    spec.loader = loader
    sys.modules[name] = module
    return module


# Lazy top-level imports: real load is deferred until first attribute access,
# but the dependency is still visible here at the top of the module.
ase = lazy_import("ase")
ase_io = lazy_import("ase.io")


class AseDataLoader(Sequence[jraph.GraphsTuple]):
    def __init__(
        self,
        path: str | Sequence[str],
        limit: int | None = None,
        read_kwargs: dict[str, Any] | None = None,
        as_graphs: dict[str, Any] | None = None,
    ):
        # Params
        self._filepath: Final[tuple[str]] = (path,) if isinstance(path, str) else tuple(path)
        self._limit: Final[int | None] = limit
        self._to_graphs: Final[dict[str, Any]] = as_graphs
        self._read_kwargs: Final[dict[str, Any]] = self._init_kwargs(limit, read_kwargs)

        try:
            loaded: list["ase.Atoms"] = []
            for entry in self._filepath:
                loaded.extend(ase_io.read(entry, **self._read_kwargs))
            self._data: list["ase.Atoms" | jraph.GraphsTuple] = (
                [loaded] if isinstance(loaded, ase.Atoms) else loaded
            )
        except FileNotFoundError:
            raise ValueError(
                f"Could not load ASE structures, the passed path does not exist: {path}"
            ) from None

        if self._to_graphs and len(self) > 0:
            # Check that the parameters they passed are OK by requesting the first structure
            # to be converted
            self[0]  # noqa, pylint: disable=pointless-statement

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, item: int) -> Any:
        entry = self._data[item]
        if self._to_graphs and not isinstance(entry, jraph.GraphsTuple):
            # Lazily convert the first time
            entry = atomic.graph_from_ase(entry, **self._to_graphs)
            self._data[item] = entry
        return entry

    @staticmethod
    def _init_kwargs(
        limit: int | None, read_kwargs: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if read_kwargs is None:
            read_kwargs = {"index": f":{limit}" if limit is not None else ":"}
        else:
            if limit is not None:
                read_kwargs["index"] = f":{limit}"
            elif "index" not in read_kwargs:
                read_kwargs["index"] = ":"
        return read_kwargs
