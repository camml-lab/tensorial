# -*- coding: utf-8 -*-
from __future__ import annotations  # For py39

import functools

import jax
import jraph
from pytray import tree

from . import _typing


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


def path_from_str(path_str: _typing.TreePathLike, delimiter=".") -> _typing.TreePath:
    """Split up a path string into a tuple of path components"""
    if isinstance(path_str, tuple):
        return path_str

    return tuple(path_str.split(delimiter))


def path_to_str(path: _typing.TreePathLike, delimiter=".") -> str:
    """Return a string representation of a tree path"""
    if isinstance(path, str):
        return path

    return delimiter.join(path)


def get(graph: jraph.GraphsTuple, *path: _typing.TreePathLike) -> jax.Array | tuple[jax.Array, ...]:
    path = tuple(map(path_from_str, path))
    graph_dict = graph._asdict()
    vals = tuple(map(functools.partial(tree.get_by_path, graph_dict), path))
    if len(path) == 1:
        return vals[0]

    return vals
