# -*- coding: utf-8 -*-
import functools

import jax

TreePath = tuple[str, ...]
TreePathLike = str | TreePath


@functools.singledispatch
def key_to_str(key) -> str:
    raise ValueError(key)


@key_to_str.register
def attr_key_to_str(key: jax.tree_util.GetAttrKey) -> str:
    return key.name


@key_to_str.register
def dict_key_to_str(key: jax.tree_util.DictKey) -> str:
    return str(key.key)


@key_to_str.register
def sequence_key_to_str(key: jax.tree_util.SequenceKey) -> str:
    return str(key.idx)


@key_to_str.register
def indexed_key_to_str(key: jax.tree_util.FlattenedIndexKey) -> str:
    return str(key.key)


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
