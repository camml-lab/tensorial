from collections.abc import Callable
from typing import Union

import jraph

__all__ = "TreePath", "TreePathLike", "GraphFunction"

TreePath = tuple[str, ...]
TreePathLike = Union[str, TreePath]

# Function that takes a graph and returns a graph
GraphFunction = Callable[[jraph.GraphsTuple], jraph.GraphsTuple]
