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

__all__ = ("grad", "jacobian", "jacrev", "jacfwd", "Grad", "Jacobian", "Jacfwd")

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
    sum_axis: Optional[Union[bool, int]] = None,
) -> Callable[[jraph.GraphsTuple, ...], tuple[..., jraph.GraphsTuple]]:
    """
    Create a function that takes the values of the quantities we want to take the derivatives with
    respect to
    """

    def shim(graph: jraph.GraphsTuple, *args) -> tuple[..., jraph.GraphsTuple]:

        new_fn = _base.transform_fn(fn, *wrt, outs=of, return_graphs=True)

        # Pass the graph through the function
        res = new_fn(graph, *args)
        if not of:
            return res

        *vals, out_graph = res

        # Extract the quantity that we want to differentiate
        def _convert(value):
            value = base.as_array(value)
            if sum_axis is not False:
                value = value.sum(axis=sum_axis)
            return value

        vals = tuple(map(_convert, vals))
        if len(vals) == 1:
            vals = vals[0]

        return vals, out_graph

    return shim


def _graph_autodiff(
    diff_fn: Callable,
    func: "gcnn.typing.GraphFunction",
    of: "gcnn.typing.TreePathLike",
    wrt: Union[str, Sequence["gcnn.typing.TreePathLike"]],
    out_field: Union[str, Sequence[str]] = "auto",
    sign: float = 1.0,
    sum_axis=None,
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
    shim = _create_grad_shim(func, of, *wrt, sum_axis=sum_axis)
    grad_fn = diff_fn(shim, argnums=tuple(range(1, len(wrt) + 1)), has_aux=True)

    # Evaluate
    def calc_grad(graph: jraph.GraphsTuple, *wrt_values) -> GradOut:
        if len(wrt_values) != len(wrt):
            raise ValueError(
                f"Failed to supply valued to evaluate derivatives at, expected: "
                f"{','.join(map(_tree.path_to_str, wrt))}"
            )

        grads, graph_out = grad_fn(graph, *wrt_values)
        if out_field is None:
            # In this case, the user just wants the raw value and does not expect a graph back
            if len(wrt_values) == 1:
                return grads[0]

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
) -> Callable[["gcnn.typing.GraphFunction"], Callable[[jraph.GraphsTuple, ...], GradOut]]:
    """
    Build a partially initialised Grad function whose only
    :param kwargs: accepts any arguments that `Grad` does
    :return: the partially initialized Grad function
    """
    return functools.partial(
        _graph_autodiff, jax.grad, of=of, wrt=wrt, out_field=out_field, sign=sign
    )


def jacrev(
    of: Union["gcnn.typing.TreePathLike", Sequence["gcnn.typing.TreePathLike"]],
    wrt: Union["gcnn.typing.TreePathLike", Sequence["gcnn.typing.TreePathLike"]],
    out_field: Optional[Union[str, Sequence[str]]] = "auto",
    sign: float = 1.0,
) -> Callable[["gcnn.typing.GraphFunction"], Callable[[jraph.GraphsTuple, ...], GradOut]]:
    """
    Build a partially initialised Grad function whose only
    :param kwargs: accepts any arguments that `Grad` does
    :return: the partially initialized Grad function
    """
    return functools.partial(
        _graph_autodiff,
        jax.jacrev,
        of=of,
        wrt=wrt,
        out_field=out_field,
        sign=sign,
        sum_axis=0,
    )


def jacfwd(
    of: Union["gcnn.typing.TreePathLike", Sequence["gcnn.typing.TreePathLike"]],
    wrt: Union["gcnn.typing.TreePathLike", Sequence["gcnn.typing.TreePathLike"]],
    out_field: Optional[Union[str, Sequence[str]]] = "auto",
    sign: float = 1.0,
) -> Callable[["gcnn.typing.GraphFunction"], Callable[[jraph.GraphsTuple, ...], GradOut]]:
    """
    Build a partially initialised Grad function whose only
    :param kwargs: accepts any arguments that `Grad` does
    :return: the partially initialized Grad function
    """
    return functools.partial(
        _graph_autodiff,
        jax.jacfwd,
        of=of,
        wrt=wrt,
        out_field=out_field,
        sign=sign,
        sum_axis=0,
    )


jacobian = jacrev


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
        if isinstance(self.wrt, str):
            wrt = [_tree.get(graph, self.wrt)]
        else:
            wrt = _tree.get(graph, *self._wrt)

        return self._grad_fn(graph, *wrt)


class Jacobian(linen.Module):
    func: "gcnn.typing.GraphFunction"
    of: "gcnn.typing.TreePathLike"
    wrt: Union[str, Sequence["gcnn.typing.TreePathLike"]]
    out_field: Union[str, Sequence[str]] = "auto"
    sign: float = 1.0

    def setup(self):
        # pylint: disable=attribute-defined-outside-init
        self._grad_fn = jacobian(self.of, self.wrt, self.out_field, self.sign)(self.func)

    @_base.shape_check
    def __call__(self, graph: jraph.GraphsTuple) -> GradOut:
        if isinstance(self.wrt, str):
            wrt = [_tree.get(graph, self.wrt)]
        else:
            wrt = _tree.get(graph, *self._wrt)

        return self._grad_fn(graph, *wrt)


class Jacfwd(linen.Module):
    func: "gcnn.typing.GraphFunction"
    of: "gcnn.typing.TreePathLike"
    wrt: Union[str, Sequence["gcnn.typing.TreePathLike"]]
    out_field: Union[str, Sequence[str]] = "auto"
    sign: float = 1.0

    def setup(self):
        # pylint: disable=attribute-defined-outside-init
        self._grad_fn = jacfwd(self.of, self.wrt, self.out_field, self.sign)(self.func)

    @_base.shape_check
    def __call__(self, graph: jraph.GraphsTuple) -> GradOut:
        if isinstance(self.wrt, str):
            wrt = [_tree.get(graph, self.wrt)]
        else:
            wrt = _tree.get(graph, *self._wrt)

        return self._grad_fn(graph, *wrt)
