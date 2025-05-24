import collections.abc
import logging
from typing import Optional

import jraph

from ._tree import path_from_str, path_to_str
from ._typing import TreePath, TreePathLike

__all__ = "UpdateDict", "TreePath", "TreePathLike", "path_from_str", "path_to_str"

_LOGGER = logging.getLogger(__name__)


class UpdateDict(collections.abc.MutableMapping):
    """
    This class can be used to make updates to a dictionary without modifying the passed dictionary.
    Once all the updates are made, a new dictionary that is the result of the modifications can be
    retrieved using the `_asdict()` method.
    """

    DELETED = tuple()  # Just a token we use to indicate a deleted value

    def __init__(self, updating: dict):
        super().__init__()
        self._updating = updating
        self._overrides = {}

    def __getitem__(self, item):
        try:
            value = self._overrides[item]
        except KeyError:
            pass
        else:
            if value is self.DELETED:
                raise KeyError(item)
            return value

        value = self._updating.__getitem__(item)
        if type(value) is dict:  # pylint: disable=unidiomatic-typecheck
            value = UpdateDict(value)
            self._overrides[item] = value

        return value

    def __setitem__(self, key, value):
        self._overrides[key] = value

    def __delitem__(self, key):
        self._overrides[key] = self.DELETED

    def __iter__(self):
        for key in self._updating.__iter__():
            try:
                value = self._overrides[key]
            except KeyError:
                yield key
            else:
                if value is not self.DELETED:
                    yield key

    def __len__(self):
        return len(self._updating) - len(
            list(filter(lambda val: val is self.DELETED, self._overrides.values()))
        )

    def _asdict(self) -> dict:
        self_dict = dict(self._updating)
        for key, value in self._overrides.items():
            if value is self.DELETED:
                del self_dict[key]
            elif isinstance(value, UpdateDict):
                self_dict[key] = value._asdict()
            else:
                self_dict[key] = value
        return self_dict


class UpdateGraphDicts:
    def __init__(self, graph: jraph.GraphsTuple):
        self._original = graph
        self._nodes: Optional[UpdateDict] = None
        self._edges: Optional[UpdateDict] = None
        self._globals: Optional[UpdateDict] = None

    @property
    def nodes(self) -> dict:
        if self._nodes is None:
            self._nodes = UpdateDict(self._original.nodes)
        return self._nodes

    @property
    def edges(self) -> dict:
        if self._edges is None:
            self._edges = UpdateDict(self._original.edges)
        return self._edges

    @property
    def globals(self) -> dict:
        if self._globals is None:
            self._globals = UpdateDict(self._original.globals)
        return self._globals

    @property
    def senders(self):
        return self._original.senders

    @property
    def receivers(self):
        return self._original.receivers

    @property
    def n_node(self):
        return self._original.n_node

    @property
    def n_edge(self):
        return self._original.n_edge

    def get(self) -> jraph.GraphsTuple:
        replacements = {}
        if self._nodes is not None:
            replacements["nodes"] = self._nodes._asdict()
        if self._edges is not None:
            replacements["edges"] = self._edges._asdict()
        if self._globals is not None:
            replacements["globals"] = self._globals._asdict()

        if not replacements:
            return self._original

        return self._original._replace(**replacements)
