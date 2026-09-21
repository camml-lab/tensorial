"""Utility helpers for the ``tensorial.gcnn`` namespace (tree paths, update dicts)."""

import collections.abc
import logging

from ._tree import path_from_str, path_to_str

__all__ = "UpdateDict", "path_from_str", "path_to_str"

_LOGGER = logging.getLogger(__name__)


class UpdateDict(collections.abc.MutableMapping):
    """This class can be used to make updates to a dictionary without modifying the passed
    dictionary. Once all the updates are made, a new dictionary that is the result of the
    modifications can be retrieved using the `_asdict()` method.
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

        # Yield new keys added to overrides
        for key, value in self._overrides.items():
            if key not in self._updating and value is not self.DELETED:
                yield key

    def __len__(self):
        count = len(self._updating)
        for key, value in self._overrides.items():
            if key in self._updating:
                if value is self.DELETED:
                    count -= 1
            else:
                if value is not self.DELETED:
                    count += 1
        return count

    def _asdict(self) -> dict:
        self_dict = dict(self._updating)
        for key, value in self._overrides.items():
            if value is self.DELETED:
                self_dict.pop(key, None)
            elif isinstance(value, UpdateDict):
                self_dict[key] = value._asdict()
            else:
                self_dict[key] = value
        return self_dict
