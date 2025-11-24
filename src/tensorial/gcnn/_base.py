from collections.abc import Sequence
import functools
import logging
from typing import TYPE_CHECKING, Protocol

from jax import tree_util
import jaxtyping as jt
import jraph

from . import _tree
from .experimental import utils as exp_utils
from .typing import GraphFunction

if TYPE_CHECKING:
    from tensorial import gcnn

__all__ = "GraphFunction", "shape_check", "adapt", "transform_fn"

_LOGGER = logging.getLogger(__name__)


def shape_check(func: "gcnn.typing.GraphFunction") -> "gcnn.typing.GraphFunction":
    """Decorator that will print to the logger any differences in either the keys present in
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


class TransformedGraphFunction(Protocol):
    """Transformed graph function that returns a value or a tuple of a value and a graph"""

    def __call__(
        self, graph: jraph.GraphsTuple, *args: jt.PyTree, **kwargs: jt.PyTree
    ) -> jt.PyTree | tuple[jt.PyTree, jraph.GraphsTuple]: ...


def adapt(
    fn: "gcnn.typing.ExGraphFunction",
    *args: "gcnn.TreePathLike",
    outs: "Sequence[gcnn.TreePathLike]" = tuple(),
    return_graphs: bool = False,
    **keywords: "gcnn.TreePathLike",
) -> TransformedGraphFunction:
    """Given a graph function, this will return a function that takes a graph as the first argument
    followed by positional arguments that will be mapped to the fields given by ``ins``.
    Output paths can optionally be specified with ``outs`` which, if supplied, will make the
    function return one or more values from the graph as returned by ``fn``.

    Args:
        fn: the graph function
        *args: the input paths
        outs: the output paths
        return_graphs: if `True` and ``outs`` is specified, this will
            return a tuple containing the output graph followed by the
            values at ``outs``

    Returns:
        a function that wraps ``fn`` with the above properties
    """
    args = tuple(_tree.path_from_str(path) for path in args)
    kwarg_paths = {name: _tree.path_to_str(path) for name, path in keywords.items()}
    outs = tuple(_tree.path_from_str(path) for path in outs)

    def _fn(
        graph: jraph.GraphsTuple, *arg_values: jt.PyTree, **kwarg_values
    ) -> jt.PyTree | tuple[jt.PyTree, jraph.GraphsTuple]:
        # Set the values from graph at the correct paths in the graphs tuple
        updater = exp_utils.update_graph(graph)
        # Update from positional arguments first
        for path, arg in zip(args, arg_values):
            updater.set(path, arg)
        # Now from kwargs
        for name, value in kwarg_values.items():
            updater.set(kwarg_paths[name], value)

        graph = updater.get()

        # Pass the graph through the original function
        res = fn(graph, *arg_values[len(args) :])
        if outs:
            # Extract the quantity that we want as outputs
            out_graph: jraph.GraphsTuple = res
            out_vals = _tree.get(out_graph, *outs)
            if return_graphs:
                return out_vals, out_graph

            return out_vals

        if return_graphs:
            # Just return the original input graph
            return res, graph

        return res

    return _fn


transform_fn = adapt  # For backward compatibility
