# -*- coding: utf-8 -*-
import collections.abc
from typing import Tuple, Union

TreePath = Tuple[str, ...]
TreePathLike = Union[TreePath, str]


def path_from_str(path_str: TreePathLike, delimiter=".") -> TreePath:
    """Split up a path string into a tuple of path components"""
    if isinstance(path_str, tuple):
        return path_str

    return tuple(path_str.split(delimiter))


def path_to_str(path: TreePathLike, delimiter=".") -> str:
    """Return a string representation of a tree path"""
    if isinstance(path, str):
        return path

    return delimiter.join(path)


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
