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

__all__ = ("grad", "jacobian", "jacrev", "jacfwd", "hessian", "Grad", "Jacobian", "Jacfwd")

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
    sign: float = 1.0,
    sum_axis=None,
    has_aux: bool = False,
) -> Callable[[jraph.GraphsTuple], GradOut]:
    # Gradient of
    of = _tree.to_paths(of)

    # Gradient with respect to
    wrt = _tree.to_paths(wrt)

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
        grads = [sign * grad for grad in grads]
        if len(wrt_values) == 1:
            grads = grads[0]

        if has_aux:
            return grads, graph_out

        return grads

    return calc_grad


def grad(
    of: Union["gcnn.typing.TreePathLike", Sequence["gcnn.typing.TreePathLike"]],
    wrt: Union["gcnn.typing.TreePathLike", Sequence["gcnn.typing.TreePathLike"]],
    sign: float = 1.0,
    has_aux: bool = False,
) -> Callable[["gcnn.typing.GraphFunction"], Callable[[jraph.GraphsTuple, ...], GradOut]]:
    """
    Build a partially initialised Grad function whose only
    :param kwargs: accepts any arguments that `Grad` does
    :return: the partially initialized Grad function
    """
    return functools.partial(_graph_autodiff, jax.grad, of=of, wrt=wrt, sign=sign, has_aux=has_aux)


def jacrev(
    of: Union["gcnn.typing.TreePathLike", Sequence["gcnn.typing.TreePathLike"]],
    wrt: Union["gcnn.typing.TreePathLike", Sequence["gcnn.typing.TreePathLike"]],
    sign: float = 1.0,
    has_aux: bool = False,
) -> Callable[["gcnn.typing.GraphFunction"], Callable[[jraph.GraphsTuple, ...], GradOut]]:
    """
    Build a partially initialised Grad function whose only
    :param kwargs: accepts any arguments that `Grad` does
    :return: the partially initialized Grad function
    """
    return functools.partial(
        _graph_autodiff, jax.jacrev, of=of, wrt=wrt, sign=sign, sum_axis=0, has_aux=has_aux
    )


def jacfwd(
    of: Union["gcnn.typing.TreePathLike", Sequence["gcnn.typing.TreePathLike"]],
    wrt: Union["gcnn.typing.TreePathLike", Sequence["gcnn.typing.TreePathLike"]],
    sign: float = 1.0,
    has_aux: bool = False,
) -> Callable[["gcnn.typing.GraphFunction"], Callable[[jraph.GraphsTuple, ...], GradOut]]:
    """
    Build a partially initialised Grad function whose only
    :param kwargs: accepts any arguments that `Grad` does
    :return: the partially initialized Grad function
    """
    return functools.partial(
        _graph_autodiff, jax.jacfwd, of=of, wrt=wrt, sign=sign, sum_axis=0, has_aux=has_aux
    )


jacobian = jacrev


def hessian(
    of: Union["gcnn.typing.TreePathLike", Sequence["gcnn.typing.TreePathLike"]],
    wrt: Union["gcnn.typing.TreePathLike", Sequence["gcnn.typing.TreePathLike"]],
    sign: float = 1.0,
    has_aux: bool = False,
) -> Callable[["gcnn.typing.GraphFunction"], Callable[[jraph.GraphsTuple, ...], GradOut]]:
    """
    Build a partially initialised Grad function whose only
    :param kwargs: accepts any arguments that `Grad` does
    :return: the partially initialized Grad function
    """
    return functools.partial(
        _graph_autodiff, jax.hessian, of=of, wrt=wrt, sign=sign, sum_axis=None, has_aux=has_aux
    )


class Grad(linen.Module):
    func: "gcnn.typing.GraphFunction"
    of: "gcnn.typing.TreePathLike"  # Gradient of
    wrt: Union[str, Sequence["gcnn.typing.TreePathLike"]]  # Gradient with respect to
    out_field: Union[str, Sequence[str]] = "auto"
    sign: float = 1.0

    def setup(self):
        # pylint: disable=attribute-defined-outside-init
        self._of = _tree.to_paths(self.of)
        self._wrt = _tree.to_paths(self.wrt)
        self._out_field = (
            _create(self._of, self._wrt) if self.out_field == "auto" else self.out_field
        )
        self._grad_fn = grad(self._of, self._wrt, sign=self.sign)(self.func)

    @_base.shape_check
    def __call__(self, graph: jraph.GraphsTuple) -> GradOut:
        wrt = _tree.get(graph, *self._wrt)
        if len(self._wrt) == 1:
            wrt = [wrt]

        res = self._grad_fn(graph, *wrt)
        graph_updates = graph._asdict()

        for field, value in zip(self._out_field, res):
            tree.set_by_path(graph_updates, field, value)

        return jraph.GraphsTuple(**graph_updates)


class Jacobian(linen.Module):
    func: "gcnn.typing.GraphFunction"
    of: "gcnn.typing.TreePathLike"
    wrt: Union[str, Sequence["gcnn.typing.TreePathLike"]]
    out_field: Union[str, Sequence[str]] = "auto"
    sign: float = 1.0

    def setup(self):
        # pylint: disable=attribute-defined-outside-init
        self._of = _tree.to_paths(self.of)
        self._wrt = _tree.to_paths(self.wrt)
        self._out_field = (
            _create(self._of, self._wrt) if self.out_field == "auto" else self.out_field
        )
        self._grad_fn = jacobian(self.of, self.wrt, self.out_field, self.sign)(self.func)

    @_base.shape_check
    def __call__(self, graph: jraph.GraphsTuple) -> GradOut:
        wrt = _tree.get(graph, *self._wrt)
        if len(self._wrt) == 1:
            wrt = [wrt]

        res = self._grad_fn(graph, *wrt)
        graph_updates = graph._asdict()

        for field, value in zip(self._out_field, res):
            tree.set_by_path(graph_updates, field, value)

        return jraph.GraphsTuple(**graph_updates)


class Jacfwd(linen.Module):
    func: "gcnn.typing.GraphFunction"
    of: "gcnn.typing.TreePathLike"
    wrt: Union[str, Sequence["gcnn.typing.TreePathLike"]]
    out_field: Union[str, Sequence[str]] = "auto"
    sign: float = 1.0

    def setup(self):
        # pylint: disable=attribute-defined-outside-init
        self._of = _tree.to_paths(self.of)
        self._wrt = _tree.to_paths(self.wrt)
        self._out_field = (
            _create(self._of, self._wrt) if self.out_field == "auto" else self.out_field
        )
        self._grad_fn = jacfwd(self.of, self.wrt, self.out_field, self.sign)(self.func)

    @_base.shape_check
    def __call__(self, graph: jraph.GraphsTuple) -> GradOut:
        wrt = _tree.get(graph, *self._wrt)
        if len(self._wrt) == 1:
            wrt = [wrt]

        res = self._grad_fn(graph, *wrt)
        graph_updates = graph._asdict()

        for field, value in zip(self._out_field, res):
            tree.set_by_path(graph_updates, field, value)

        return jraph.GraphsTuple(**graph_updates)


def _create(of: Sequence[tuple], wrt: Sequence[tuple]) -> list[tuple]:
    derivs = []
    for wrt_entry in wrt:
        for of_entry in of:
            derivs.append(wrt_entry[:-1] + (f"d{'.'.join(of_entry[1:])}/d{wrt_entry[-1]}",))

    return derivs
