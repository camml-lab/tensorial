from collections.abc import Sequence
import functools
from typing import TYPE_CHECKING, Final, Literal

import jax
from jaxtyping import Array
import jraph
from pytray import tree

from . import keys

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
    if path_str == "":
        return tuple()

    return tuple(path_str.split(delimiter))


def path_to_str(path: "gcnn.typing.TreePathLike", delimiter: str = DEFAULT_DELIMITER) -> str:
    """Return a string representation of a tree path"""
    if isinstance(path, str):
        return path

    return delimiter.join(path)


def get(
    graph: jraph.GraphsTuple, *path: "gcnn.typing.TreePathLike"
) -> jax.Array | tuple[jax.Array, ...]:
    """Given a graph, this will extract the values as the passed path(s) and return them directly

    Args:
        graph: the graph to get values from
        *path: the path(s)

    Returns:
        the values at those paths
    """
    path = tuple(map(path_from_str, path))
    graph_dict = graph._asdict()
    vals = tuple(map(functools.partial(tree.get_by_path, graph_dict), path))
    if len(path) == 1:
        return vals[0]

    return vals


def to_paths(
    wrt: str | Sequence["gcnn.typing.TreePathLike"] | None,
) -> "tuple[gcnn.typing.TreePath, ...]":
    """Normalise a path specification into a tuple of tree paths.

    Converts strings, sequences of paths, or None into a consistent
    tuple of path tuples for use throughout the library.

    Args:
        wrt: A string path, sequence of paths, or None.

    Returns:
        A tuple of parsed tree paths.

    Raises:
        ValueError: If `wrt` has an unsupported type.
    """
    if wrt is None:
        return tuple()
    if isinstance(wrt, str):
        return (path_from_str(wrt),)
    if isinstance(wrt, Sequence):
        return tuple(map(path_from_str, wrt))

    raise ValueError(f"wrt must be str or list or tuple thereof, got {type(wrt).__name__}")


def path_root(
    path: "gcnn.typing.TreePathLike",
    delimiter=DEFAULT_DELIMITER,
) -> "gcnn.typing.TreePath":
    """Return the root component (first element) of a tree path."""
    return path_from_str(path, delimiter=delimiter)[:1]


def get_mask(
    graph: jraph.GraphsTuple,
    path: "gcnn.typing.TreePathLike",
    delimiter=DEFAULT_DELIMITER,
) -> jax.Array | None:
    """Get the mask for a given path in the graph, if it exists.

    Constructs the path to the mask by appending 'mask' to the root of the given path,
    then retrieves the value from the graph.

    Args:
        graph: The graph to get the mask from
        path: The tree path whose root determines which component's mask to get
        delimiter: The delimiter used to parse the path

    Returns:
        The mask array if found, or None if the path doesn't exist
    """
    path = path_root(path, delimiter) + (keys.MASK,)
    try:
        return tree.get_by_path(graph._asdict(), path)
    except KeyError:
        return None


def num_entries(entity: Literal["nodes", "edges", "globals"], graph: jraph.GraphsTuple) -> int:
    """Return the number of entries in a component of the graph

    Args:
        entity: The graph component to count entries in ('nodes', 'edges', or 'globals')
        graph: The graph to count entries in

    Returns:
        The number of entries in the specified component
    """
    if entity not in ("nodes", "edges", "globals"):
        raise ValueError(f"entity must be one of 'nodes', 'edges', or 'globals', got {entity}")

    if entity == "globals":
        return len(graph.n_node)

    entries: dict[str, Array] = getattr(graph, entity)
    return next(iter(entries.values())).shape[0] if entries else 0
