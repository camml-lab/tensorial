from collections.abc import Callable
from typing import Protocol, Union

import jaxtyping as jt
import jraph

__all__ = "TreePath", "TreePathLike", "GraphFunction"


TreePath = tuple[str, ...]
TreePathLike = Union[str, TreePath]

# Function that takes a graph and returns a graph
GraphFunction = Callable[[jraph.GraphsTuple], jraph.GraphsTuple]


class ExGraphFunction(Protocol):
    """Extended graph function that can optionally take a graph and additional arguments"""

    def __call__(self, graph: jraph.GraphsTuple, *args: jt.PyTree) -> jraph.GraphsTuple: ...
