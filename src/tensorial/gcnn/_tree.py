from collections.abc import Sequence
import functools
from typing import TYPE_CHECKING, Final, Optional, Union

import jax
import jraph
from pytray import tree

if TYPE_CHECKING:
    from tensorial import gcnn

DEFAULT_DELIMITER: Final[str] = "."


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


def path_from_str(
    path_str: "gcnn.typing.TreePathLike", delimiter: str = DEFAULT_DELIMITER
) -> "gcnn.typing.TreePath":
    """Split up a path string into a tuple of path components"""
    if isinstance(path_str, tuple):
        return path_str

    return tuple(path_str.split(delimiter))


def path_to_str(path: "gcnn.typing.TreePathLike", delimiter: str = DEFAULT_DELIMITER) -> str:
    """Return a string representation of a tree path"""
    if isinstance(path, str):
        return path

    return delimiter.join(path)


def get(
    graph: jraph.GraphsTuple, *path: "gcnn.typing.TreePathLike"
) -> Union[jax.Array, tuple[jax.Array, ...]]:
    """
    Given a graph, this will extract the values as the passed path(s) and return them directly

    :param graph: the graph to get values from
    :param path: the path(s)
    :return: the values at those paths
    """
    path = tuple(map(path_from_str, path))
    graph_dict = graph._asdict()
    vals = tuple(map(functools.partial(tree.get_by_path, graph_dict), path))
    if len(path) == 1:
        return vals[0]

    return vals


def to_paths(
    wrt: Optional[Union[str, Sequence["gcnn.typing.TreePathLike"]]],
) -> "tuple[gcnn.typing.TreePath, ...]":
    if wrt is None:
        return tuple()
    if isinstance(wrt, str):
        return (path_from_str(wrt),)
    if isinstance(wrt, Sequence):
        return tuple(map(path_from_str, wrt))

    raise ValueError(f"wrt must be str or list or tuple thereof, got {type(wrt).__name__}")
