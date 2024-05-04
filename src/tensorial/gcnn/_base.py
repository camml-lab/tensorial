# -*- coding: utf-8 -*-
from typing import Callable

import jraph

__all__ = ("GraphFunction",)

# Function that takes a graph and returns a graph
GraphFunction = Callable[[jraph.GraphsTuple], jraph.GraphsTuple]
