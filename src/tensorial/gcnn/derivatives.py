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
    import tensorial
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
    of: "gcnn.TreePathLike",
    wrt: "Sequence[gcnn.typing.TreePathLike]",
    sum_axis: Optional[Union[bool, int]] = None,
) -> "Callable[[jraph.GraphsTuple, ...], tuple[tensorial.typing.ArrayType, jraph.GraphsTuple]]":
    """
    Create a function that takes the values of the quantities we want to take the derivatives with
    respect to
    """
    if of is not None and len(of) < 2:
        raise ValueError(f"of must be of at least length two e.g. ('globals', 'entry'), got: {of}")

    def shim(
        graph: jraph.GraphsTuple, *args
    ) -> "tuple[tensorial.typing.ArrayType, jraph.GraphsTuple]":
        new_fn = _base.transform_fn(fn, *wrt, outs=[of], return_graphs=True)

        # Pass the graph through the function
        value, graph_out = new_fn(graph, *args)
        value = base.as_array(value)
        if sum_axis is not False:
            value = value.sum(axis=sum_axis)

        return value, graph_out

    return shim


def _graph_autodiff(
    diff_fn: Callable,
    func: "gcnn.typing.GraphFunction",
    of: "gcnn.typing.TreePathLike",
    wrt: "Union[str, Sequence[gcnn.typing.TreePathLike]]",
    sign: float = 1.0,
    sum_axis=None,
    has_aux: bool = False,
) -> Callable[[jraph.GraphsTuple], GradOut]:
    # Gradient of
    of = _tree.path_from_str(of)

    # Gradient with respect to
    wrt: tuple[gcnn.TreePath, ...] = _tree.to_paths(wrt)

    # Creat the shim which will be a function that takes the graph as first argument, and
    # the remaining values are the values to take the gradient at
    shim = _create_grad_shim(func, of, wrt, sum_axis=sum_axis)
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
    of: "gcnn.TreePathLike",
    wrt: Union["gcnn.TreePathLike", Sequence["gcnn.TreePathLike"]],
    sign: float = 1.0,
    has_aux: bool = False,
) -> Callable[["gcnn.GraphFunction"], Callable[[jraph.GraphsTuple, ...], GradOut]]:
    """
    Build a partially initialised Grad function whose only
    :param kwargs: accepts any arguments that `Grad` does
    :return: the partially initialized Grad function
    """
    return functools.partial(_graph_autodiff, jax.grad, of=of, wrt=wrt, sign=sign, has_aux=has_aux)


def jacrev(
    of: "gcnn.TreePathLike",
    wrt: Union["gcnn.TreePathLike", Sequence["gcnn.TreePathLike"]],
    sign: float = 1.0,
    has_aux: bool = False,
) -> Callable[["gcnn.GraphFunction"], Callable[[jraph.GraphsTuple, ...], GradOut]]:
    """
    Build a partially initialised Grad function whose only
    :param kwargs: accepts any arguments that `Grad` does
    :return: the partially initialized Grad function
    """
    return functools.partial(
        _graph_autodiff, jax.jacrev, of=of, wrt=wrt, sign=sign, sum_axis=0, has_aux=has_aux
    )


def jacfwd(
    of: "gcnn.TreePathLike",
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
    of: "gcnn.TreePathLike",
    wrt: Union["gcnn.TreePathLike", Sequence["gcnn.TreePathLike"]],
    sign: float = 1.0,
    has_aux: bool = False,
) -> Callable[["gcnn.GraphFunction"], Callable[[jraph.GraphsTuple, ...], GradOut]]:
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
    wrt: "Union[gcnn.TreePathLike, list[gcnn.TreePathLike]]"  # Gradient with respect to
    out_field: "Union[str, gcnn.TreePathLike, list[gcnn.TreePathLike]]" = "auto"
    sign: float = 1.0

    def setup(self):
        # pylint: disable=attribute-defined-outside-init
        self._of = _tree.to_paths(self.of)[0]
        self._wrt = _tree.to_paths(self.wrt)
        self._out_field = _out_derivative_keys(self._of, self._wrt, self.out_field)
        self._grad_fn = grad(self._of, self._wrt, sign=self.sign, has_aux=True)(self.func)

    @_base.shape_check
    def __call__(self, graph: jraph.GraphsTuple) -> GradOut:
        wrt = _tree.get(graph, *self._wrt)
        if len(self._wrt) == 1:
            wrt = [wrt]

        res, out_graph = self._grad_fn(graph, *wrt)
        if len(self._wrt) == 1:
            res = [res]

        graph_updates = out_graph._asdict()
        for path, value in zip(self._out_field, res):
            tree.set_by_path(graph_updates, path, value)

        return jraph.GraphsTuple(**graph_updates)


class Jacobian(linen.Module):
    func: "gcnn.typing.GraphFunction"
    of: "gcnn.typing.TreePathLike"
    wrt: Union[str, Sequence["gcnn.typing.TreePathLike"]]
    out_field: Union[str, Sequence[str]] = "auto"
    sign: float = 1.0

    def setup(self):
        # pylint: disable=attribute-defined-outside-init
        self._of = _tree.to_paths(self.of)[0]
        self._wrt = _tree.to_paths(self.wrt)
        self._out_field = _out_derivative_keys(self._of, self._wrt, self.out_field)
        self._grad_fn = jacobian(self.of, self.wrt, self.sign)(self.func)

    @_base.shape_check
    def __call__(self, graph: jraph.GraphsTuple) -> GradOut:
        wrt = _tree.get(graph, *self._wrt)
        if len(self._wrt) == 1:
            wrt = [wrt]

        res = self._grad_fn(graph, *wrt)
        graph_updates = graph._asdict()

        for path, value in zip(self._out_field, res):
            tree.set_by_path(graph_updates, path, value)

        return jraph.GraphsTuple(**graph_updates)


class Jacfwd(linen.Module):
    func: "gcnn.typing.GraphFunction"
    of: "gcnn.typing.TreePathLike"
    wrt: Union[str, Sequence["gcnn.typing.TreePathLike"]]
    out_field: Union[str, Sequence[str]] = "auto"
    sign: float = 1.0

    def setup(self):
        # pylint: disable=attribute-defined-outside-init
        self._of = _tree.to_paths(self.of)[0]
        self._wrt = _tree.to_paths(self.wrt)
        self._out_field = _out_derivative_keys(self._of, self._wrt, self.out_field)
        self._grad_fn = jacfwd(self.of, self.wrt, self.sign)(self.func)

    @_base.shape_check
    def __call__(self, graph: jraph.GraphsTuple) -> GradOut:
        wrt = _tree.get(graph, *self._wrt)
        if len(self._wrt) == 1:
            wrt = [wrt]

        res = self._grad_fn(graph, *wrt)
        graph_updates = graph._asdict()

        for path, value in zip(self._out_field, res):
            tree.set_by_path(graph_updates, path, value)

        return jraph.GraphsTuple(**graph_updates)


def _out_derivative_keys(
    of: "gcnn.TreePath", wrt: "Sequence[gcnn.TreePath]", out_key
) -> "tuple[gcnn.TreePath, ...]":
    if out_key == "auto":
        derivs = []
        for wrt_entry in wrt:
            derivs.append(wrt_entry[:-1] + (f"d{'.'.join(of[1:])}/d{wrt_entry[-1]}",))

        return tuple(derivs)

    if not isinstance(out_key, list):
        out_key = [out_key]

    return _tree.to_paths(out_key)


def _create(of: "gcnn.TreePath", wrt: Sequence[tuple]) -> "list[gcnn.TreePath]":
    derivs = []
    for wrt_entry in wrt:
        derivs.append(wrt_entry[:-1] + (f"d{'.'.join(of[1:])}/d{wrt_entry[-1]}",))

    return derivs
