# -*- coding: utf-8 -*-
from __future__ import annotations  # For py39

import functools
import logging
from typing import Callable, Sequence

import jax
from jax import tree_util
import jraph

from . import _tree, _typing
from ._typing import GraphFunction

__all__ = ("GraphFunction", "shape_check", "transform_fn")

_LOGGER = logging.getLogger(__name__)


def shape_check(func: _typing.GraphFunction) -> _typing.GraphFunction:
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


def transform_fn(
    fn: _typing.GraphFunction,
    *ins: _typing.TreePathLike,
    outs: Sequence[_typing.TreePathLike] = tuple(),
    return_graphs: bool = False,
) -> Callable[[jraph.GraphsTuple], ...] | Callable[[jraph.GraphsTuple, ...], ...]:
    ins = tuple(_tree.path_from_str(path) for path in ins)
    outs = tuple(_tree.path_from_str(path) for path in outs)

    def _fn(graph: jraph.GraphsTuple, *args):
        def repl(path, val):
            try:
                idx = ins.index(tuple(map(_tree.key_to_str, path)))
                return args[idx]
            except ValueError:
                return val

        # Recreate the graph using the passed values
        graph = jax.tree_util.tree_map_with_path(repl, graph)

        # Pass the graph through the original function
        out_graph = fn(graph)
        if not outs:
            return out_graph

        # Extract the quantity that we want as outputs
        vals = _tree.get(out_graph, *outs)

        if return_graphs:
            return vals, out_graph
        return vals

    return _fn
