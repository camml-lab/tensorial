from collections.abc import Callable
from typing import Any, Sequence, Union

import beartype
from flax import linen
import jax
import jax.numpy as jnp
import jaxtyping as jt
import jraph
from pytray import tree

import tensorial

from . import _base, _tree, _typing

__all__ = ("Grad",)

TreePath = tuple[Any, ...]


class Grad(linen.Module):
    func: _typing.GraphFunction
    of: _typing.TreePathLike
    wrt: Union[str, Sequence[_typing.TreePathLike]]
    out_field: Union[str, Sequence[str]] = None
    sign: float = 1.0

    def setup(self):
        # pylint: disable=attribute-defined-outside-init
        if isinstance(self.wrt, str):
            self._wrt = (_tree.path_from_str(self.wrt),)
        elif isinstance(self.wrt, Sequence):
            self._wrt = tuple(map(_tree.path_from_str, self.wrt))
        else:
            raise ValueError(
                f"wrt must be str or list or tuple thereof, got {type(self.wrt).__name__}"
            )
        self._of = _tree.path_from_str(self.of)

        if self.out_field is None:
            derivs = []
            for wrt in self._wrt:
                derivs.append(self._of[:-1] + (f"d{'.'.join(self._of[1:])}/d{wrt[-1]}",))
            self._out_field = tuple(derivs)
        else:
            self._out_field = [_tree.path_from_str(self.out_field)]

        # Creat the shim which will be a function that takes the graph as first argument, and
        # the remaining values are the values to take the gradient at
        shim = _create_grad_shim(self.func, self._of, *self._wrt)
        self._grad_fn = jax.grad(shim, argnums=tuple(range(1, len(self._wrt) + 1)), has_aux=True)

    @_base.shape_check
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        wrt_values = _tree.get(graph, *self._wrt)
        if len(self._wrt) == 1:
            wrt_values = (wrt_values,)

        grads, graph_out = self._grad_fn(graph, *wrt_values)

        # Add the gradient quantity to the output graph
        out_graph_dict = graph_out._asdict()
        for out_path, value in zip(self._out_field, grads):
            tree.set_by_path(out_graph_dict, out_path, self.sign * value)
        return jraph.GraphsTuple(**out_graph_dict)


@jt.jaxtyped(typechecker=beartype.beartype)
def grad_shim(
    fn: _typing.GraphFunction,
    graph: jraph.GraphsTuple,
    of: tuple,
    paths: tuple[_typing.TreePath],
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
    return jnp.sum(tensorial.as_array(tree.get_by_path(out_graph._asdict(), of))), out_graph


def _create_grad_shim(
    fn: _typing.GraphFunction,
    of: _typing.TreePathLike,
    *wrt: _typing.TreePathLike,
) -> Callable[[jraph.GraphsTuple, ...], tuple[..., jraph.GraphsTuple]]:
    def shim(graph: jraph.GraphsTuple, *args) -> tuple[..., jraph.GraphsTuple]:
        # Create a function that takes the values of the quantities we want to take the derivatives
        # with respect to
        new_fn = _base.transform_fn(fn, *wrt, outs=[of], return_graphs=True)

        # Pass the graph through the function
        *vals, out_graph = new_fn(graph, *args)

        # Extract the quantity that we want to differentiate
        vals = tuple(map(lambda x: tensorial.as_array(x).sum(), vals))
        if len(vals) == 1:
            vals = vals[0]

        return vals, out_graph

    return shim
