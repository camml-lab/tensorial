# -*- coding: utf-8 -*-
from __future__ import annotations  # For py39

import functools
import logging
from typing import Callable

from jax import tree_util
import jraph

from . import _tree

__all__ = ("GraphFunction", "shape_check")

_LOGGER = logging.getLogger(__name__)

# Function that takes a graph and returns a graph
GraphFunction = Callable[[jraph.GraphsTuple], jraph.GraphsTuple]


def shape_check(func: GraphFunction) -> GraphFunction:
    """
    Decorator that will print to the logger any differences in either the keys present in
    the graph before and after the call, or any differences in their shapes.

    This is super useful for diagnosing jax re-compilation issues.
    """

    @functools.wraps(func)
    def shape_checker(*args) -> jraph.GraphsTuple:
        # Can either be a class method or a free function
        inputs: jraph.GraphsTuple = args[0] if len(args) == 1 else args[1]
        flattened, _ = tree_util.tree_flatten_with_path(inputs)
        in_shapes = {path: array.shape for path, array in flattened}

        out = func(*args)
        out_shapes = {
            (path, array.shape) for path, array in tree_util.tree_flatten_with_path(out)[0]
        }
        diff = out_shapes - set(in_shapes.items())

        messages: list[str] = []
        for path, shape in diff:
            path_str = _tree.path_to_str(tuple(map(_tree.key_to_str, path)))
            try:
                in_shape = in_shapes[path]
            except KeyError:
                messages.append(f"new {path_str}")
            else:
                messages.append(f"{path_str} {in_shape}->{shape}")
        if messages:
            _LOGGER.debug(
                "%s() difference(s) in inputs/outputs: %s",
                func.__qualname__,
                ", ".join(messages),
            )

        return out

    return shape_checker
