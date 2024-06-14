# -*- coding: utf-8 -*-
from typing import Callable, Union

import jraph

__all__ = "TreePath", "TreePathLike"

TreePath = tuple[str, ...]
TreePathLike = Union[str, TreePath]

# Function that takes a graph and returns a graph
GraphFunction = Callable[[jraph.GraphsTuple], jraph.GraphsTuple]
