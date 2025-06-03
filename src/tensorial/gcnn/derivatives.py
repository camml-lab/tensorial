from collections.abc import Callable, Sequence
import functools
from typing import TYPE_CHECKING, Any, Optional, Union

import beartype
from flax import linen
import jax
import jax.numpy as jnp
import jaxtyping as jt
import jraph
from pytray import tree

from . import _base, _tree
from .. import base

if TYPE_CHECKING:
    from tensorial import gcnn

__all__ = ("Grad", "grad")

TreePath = tuple[Any, ...]

GradOut = Union[jraph.GraphsTuple, jt.PyTree, tuple[jt.PyTree]]


@jt.jaxtyped(typechecker=beartype.beartype)
def grad_shim(
    fn: "gcnn.typing.GraphFunction",
    graph: jraph.GraphsTuple,
    of: tuple,
    paths: tuple["gcnn.typing.TreePathLike"],
    *wrt_variables,
) -> tuple[jax.Array, jraph.GraphsTuple]:
    def repl(path, val):
        try:
            idx = paths.index(tuple(map(_tree.key_to_str, path)))
            return wrt_variables[idx]
        except ValueError:
            return val

    graph = jax.tree_util.tree_map_with_path(repl, graph)

    # Create the graph, now containing what the original graph plus the variables we were passed
    # graph = jraph.GraphsTuple(**graph_dict)
    # Pass the graph through the original function
    out_graph = fn(graph)
    # Extract the quantity that we want to differentiate
    return jnp.sum(base.as_array(tree.get_by_path(out_graph._asdict(), of))), out_graph


def _create_grad_shim(
    fn: "gcnn.typing.GraphFunction",
    of: Sequence["gcnn.typing.TreePathLike"],
    *wrt: "gcnn.typing.TreePathLike",
) -> Callable[[jraph.GraphsTuple, ...], tuple[..., jraph.GraphsTuple]]:
    def shim(graph: jraph.GraphsTuple, *args) -> tuple[..., jraph.GraphsTuple]:
        # Create a function that takes the values of the quantities we want to take the derivatives
        # with respect to
        new_fn = _base.transform_fn(fn, *wrt, outs=of, return_graphs=True)

        # Pass the graph through the function
        res = new_fn(graph, *args)
        if not of:
            return res

        *vals, out_graph = res

        # Extract the quantity that we want to differentiate
        vals = tuple(map(lambda x: base.as_array(x).sum(), vals))
        if len(vals) == 1:
            vals = vals[0]

        return vals, out_graph

    return shim


def _graph_grad(
    func: "gcnn.typing.GraphFunction",
    of: "gcnn.typing.TreePathLike",
    wrt: Union[str, Sequence["gcnn.typing.TreePathLike"]],
    out_field: Union[str, Sequence[str]] = "auto",
    sign: float = 1.0,
) -> Callable[[jraph.GraphsTuple], GradOut]:
    # Gradient of
    of = _tree.to_paths(of)

    # Gradient with respect to
    wrt = _tree.to_paths(wrt)

    # Save to
    if out_field is not None:
        if out_field == "auto":
            derivs = []
            for wrt_entry in wrt:
                for of_entry in of:
                    derivs.append(wrt_entry[:-1] + (f"d{'.'.join(of_entry[1:])}/d{wrt_entry[-1]}",))
            out_field = tuple(derivs)
        else:
            out_field = [_tree.path_from_str(out_field)]

    # Creat the shim which will be a function that takes the graph as first argument, and
    # the remaining values are the values to take the gradient at
    shim = _create_grad_shim(func, of, *wrt)
    grad_fn = jax.grad(shim, argnums=tuple(range(1, len(wrt) + 1)), has_aux=True)

    # Evaluate
    def calc_grad(graph: jraph.GraphsTuple) -> GradOut:
        wrt_values = _tree.get(graph, *wrt)
        if len(wrt) == 1:
            wrt_values = (wrt_values,)

        grads, graph_out = grad_fn(graph, *wrt_values)
        if out_field is None:
            # In this case, the user just wants the raw value and does not expect a graph back
            return grads

        # Add the gradient quantity to the output graph
        out_graph_dict = graph_out._asdict()
        for out_path, value in zip(out_field, grads):
            tree.set_by_path(out_graph_dict, out_path, sign * value)

        return jraph.GraphsTuple(**out_graph_dict)

    return calc_grad


def grad(
    of: Union["gcnn.typing.TreePathLike", Sequence["gcnn.typing.TreePathLike"]],
    wrt: Union["gcnn.typing.TreePathLike", Sequence["gcnn.typing.TreePathLike"]],
    out_field: Optional[Union[str, Sequence[str]]] = "auto",
    sign: float = 1.0,
) -> Callable[["gcnn.typing.GraphFunction"], Callable[[jraph.GraphsTuple], GradOut]]:
    """
    Build a partially initialised Grad function whose only
    :param kwargs: accepts any arguments that `Grad` does
    :return: the partially initialized Grad function
    """
    return functools.partial(_graph_grad, of=of, wrt=wrt, out_field=out_field, sign=sign)


class Grad(linen.Module):
    func: "gcnn.typing.GraphFunction"
    of: "gcnn.typing.TreePathLike"
    wrt: Union[str, Sequence["gcnn.typing.TreePathLike"]]
    out_field: Union[str, Sequence[str]] = "auto"
    sign: float = 1.0

    def setup(self):
        # pylint: disable=attribute-defined-outside-init
        self._grad_fn = grad(self.of, self.wrt, self.out_field, self.sign)(self.func)

    @_base.shape_check
    def __call__(self, graph: jraph.GraphsTuple) -> GradOut:
        return self._grad_fn(graph)
