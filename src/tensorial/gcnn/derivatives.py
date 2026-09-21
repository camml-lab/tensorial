"""Automatic differentiation of functions on `jraph.GraphsTuple` graphs.

Builds on JAX's ``grad`` / ``jacrev`` / ``jacfwd`` / ``hessian`` to differentiate
graph-level quantities (e.g. node/edge/global attributes) with respect to other
graph fields. Exposes both function-based factories (`grad`, `jacobian`, `jacfwd`,
`hessian`) and flax `Module` wrappers (`Grad`, `Jacobian`, `Jacfwd`) that write the
result back into the graph.
"""

from collections.abc import Callable, Sequence
import functools
from typing import TYPE_CHECKING, Any

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

GradOut = jraph.GraphsTuple | jt.PyTree | tuple[jt.PyTree]


@jt.jaxtyped(typechecker=beartype.beartype)
def grad_shim(
    fn: "gcnn.typing.GraphFunction",
    graph: jraph.GraphsTuple,
    of: tuple,
    paths: tuple["gcnn.typing.TreePathLike"],
    *wrt_variables,
) -> tuple[jax.Array, jraph.GraphsTuple]:
    """Substitute ``wrt_variables`` for the ``paths`` in the graph, then run ``fn``.

    This is a helper used by the factory functions (`grad`, `jacobian`, etc.) to
    feed the quantities being differentiated as explicit arguments into the graph
    function, so JAX can differentiate through them.

    Args:
        fn: The graph function to evaluate.
        graph: The input graph to apply the substitution to.
        of: Path to the graph attribute to differentiate (its sum).
        paths: Paths in the graph to override with ``wrt_variables``.
        wrt_variables: Values substituted for the entries of ``paths``.

    Returns:
        A tuple of ``(summed_of_value, out_graph)`` suitable for JAX autodiff.
    """

    def repl(path, val):
        """Swap in the supplied ``wrt_variables`` for the matching path."""
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
    sum_axis: bool | int | None = None,
) -> "Callable[[jraph.GraphsTuple, ...], tuple[tensorial.typing.ArrayType, jraph.GraphsTuple]]":
    """Create a function that takes the values of the quantities we want to take the derivatives
    with respect to
    """
    if of is not None and len(of) < 2:
        raise ValueError(f"of must be of at least length two e.g. ('globals', 'entry'), got: {of}")

    def shim(
        graph: jraph.GraphsTuple, *args
    ) -> "tuple[tensorial.typing.ArrayType, jraph.GraphsTuple]":
        """Run the transformed function and return ``(summed_output, out_graph)``."""
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
    wrt: "str | Sequence[gcnn.typing.TreePathLike]",
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
        """Evaluate the derivative of ``func`` with respect to the ``wrt`` fields."""
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
    wrt: "gcnn.TreePathLike | Sequence[gcnn.TreePathLike]",
    sign: float = 1.0,
    has_aux: bool = False,
) -> Callable[["gcnn.GraphFunction"], Callable[[jraph.GraphsTuple, ...], GradOut]]:
    """Build a `Grad`-style graph autograd function using JAX's ``grad``.

    Args:
        of: Path to the graph attribute to differentiate.
        wrt: Path (or list of paths) to the graph attribute(s) to differentiate through.
        sign: Multiplier applied to the resulting gradient.
        has_aux: If true, also return the output graph as an auxiliary value.

    Returns:
        A decorator that, given a graph function, returns a callable that produces
        gradient(s) with respect to ``wrt``.
    """
    return functools.partial(_graph_autodiff, jax.grad, of=of, wrt=wrt, sign=sign, has_aux=has_aux)


def jacrev(
    of: "gcnn.TreePathLike",
    wrt: "gcnn.TreePathLike | Sequence[gcnn.TreePathLike]",
    sign: float = 1.0,
    has_aux: bool = False,
) -> Callable[["gcnn.GraphFunction"], Callable[[jraph.GraphsTuple, ...], GradOut]]:
    """Build a `Jacobian`-style graph autograd function using JAX's ``jacrev``.

    Args:
        of: Path to the graph attribute to differentiate.
        wrt: Path (or list of paths) to the graph attribute(s) to differentiate through.
        sign: Multiplier applied to the resulting Jacobian.
        has_aux: If true, also return the output graph as an auxiliary value.

    Returns:
        A decorator that, given a graph function, returns a callable that produces
        Jacobian(s) with respect to ``wrt``.
    """
    return functools.partial(
        _graph_autodiff, jax.jacrev, of=of, wrt=wrt, sign=sign, sum_axis=0, has_aux=has_aux
    )


def jacfwd(
    of: "gcnn.TreePathLike",
    wrt: "gcnn.typing.TreePathLike | Sequence[gcnn.typing.TreePathLike]",
    sign: float = 1.0,
    has_aux: bool = False,
) -> Callable[["gcnn.typing.GraphFunction"], Callable[[jraph.GraphsTuple, ...], GradOut]]:
    """Build a `Jacfwd`-style graph autograd function using JAX's ``jacfwd``.

    Args:
        of: Path to the graph attribute to differentiate.
        wrt: Path (or list of paths) to the graph attribute(s) to differentiate through.
        sign: Multiplier applied to the resulting Jacobian.
        has_aux: If true, also return the output graph as an auxiliary value.

    Returns:
        A decorator that, given a graph function, returns a callable that produces
        Jacobian(s) with respect to ``wrt``.
    """
    return functools.partial(
        _graph_autodiff, jax.jacfwd, of=of, wrt=wrt, sign=sign, sum_axis=0, has_aux=has_aux
    )


jacobian = jacrev


def hessian(
    of: "gcnn.TreePathLike",
    wrt: "gcnn.TreePathLike | Sequence[gcnn.TreePathLike]",
    sign: float = 1.0,
    has_aux: bool = False,
) -> Callable[["gcnn.GraphFunction"], Callable[[jraph.GraphsTuple, ...], GradOut]]:
    """Build a `Hessian`-style graph autograd function using JAX's ``hessian``.

    Args:
        of: Path to the graph attribute to differentiate.
        wrt: Path (or list of paths) to the graph attribute(s) to differentiate through.
        sign: Multiplier applied to the resulting Hessian.
        has_aux: If true, also return the output graph as an auxiliary value.

    Returns:
        A decorator that, given a graph function, returns a callable that produces
        Hessian(s) with respect to ``wrt``.
    """
    return functools.partial(
        _graph_autodiff, jax.hessian, of=of, wrt=wrt, sign=sign, sum_axis=None, has_aux=has_aux
    )


class Grad(linen.Module):
    """
    The `Grad` class computes gradients of graph-based functions with respect to specified graph
    attributes. It enables automatic differentiation of operations defined on graph structures,
    such as computing how changes in node positions affect edge lengths or other graph properties.
    The class supports both scalar and vector-valued gradients and integrates with JAX for efficient
    computation.
    """

    func: "gcnn.typing.GraphFunction"
    of: "gcnn.typing.TreePathLike"  # Gradient of
    wrt: "gcnn.TreePathLike | list[gcnn.TreePathLike]"  # Gradient with respect to
    out_field: "str | gcnn.TreePathLike | list[gcnn.TreePathLike]" = "auto"
    sign: float = 1.0

    def setup(self):
        """Resolve the of/wrt paths and build the underlying JAX gradient function."""
        # pylint: disable=attribute-defined-outside-init
        self._of = _tree.to_paths(self.of)[0]
        self._wrt = _tree.to_paths(self.wrt)
        self._out_field = _out_derivative_keys(self._of, self._wrt, self.out_field)
        self._grad_fn = grad(self._of, self._wrt, sign=self.sign, has_aux=True)(self.func)

    @_base.shape_check
    def __call__(self, graph: jraph.GraphsTuple) -> GradOut:
        """Compute the gradient and write it back into the graph under ``out_field``."""
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
    """Compute the reverse-mode Jacobian of a graph field with respect to another.

    Like `Grad`, but returns the full Jacobian (``jacfwd``/``jacrev``) instead of a
    scalar gradient, writing the result back into the graph under ``out_field``.
    """

    func: "gcnn.typing.GraphFunction"
    of: "gcnn.typing.TreePathLike"
    wrt: str | Sequence["gcnn.typing.TreePathLike"]
    out_field: str | Sequence[str] = "auto"
    sign: float = 1.0

    def setup(self):
        """Resolve the of/wrt paths and build the underlying JAX Jacobian function."""
        # pylint: disable=attribute-defined-outside-init
        self._of = _tree.to_paths(self.of)[0]
        self._wrt = _tree.to_paths(self.wrt)
        self._out_field = _out_derivative_keys(self._of, self._wrt, self.out_field)
        self._grad_fn = jacobian(self.of, self.wrt, self.sign)(self.func)

    @_base.shape_check
    def __call__(self, graph: jraph.GraphsTuple) -> GradOut:
        """Compute the Jacobian and write it back into the graph under ``out_field``."""
        wrt = _tree.get(graph, *self._wrt)
        if len(self._wrt) == 1:
            wrt = [wrt]

        res = self._grad_fn(graph, *wrt)
        graph_updates = graph._asdict()

        for path, value in zip(self._out_field, res):
            tree.set_by_path(graph_updates, path, value)

        return jraph.GraphsTuple(**graph_updates)


class Jacfwd(linen.Module):
    """Compute the forward-mode Jacobian of a graph field with respect to another.

    Like `Jacobian`, but uses JAX's forward-mode ``jacfwd`` rule, writing the result
    back into the graph under ``out_field``.
    """

    func: "gcnn.typing.GraphFunction"
    of: "gcnn.typing.TreePathLike"
    wrt: str | Sequence["gcnn.typing.TreePathLike"]
    out_field: str | Sequence[str] = "auto"
    sign: float = 1.0

    def setup(self):
        """Resolve the of/wrt paths and build the underlying JAX Jacobian function."""
        # pylint: disable=attribute-defined-outside-init
        self._of = _tree.to_paths(self.of)[0]
        self._wrt = _tree.to_paths(self.wrt)
        self._out_field = _out_derivative_keys(self._of, self._wrt, self.out_field)
        self._grad_fn = jacfwd(self.of, self.wrt, self.sign)(self.func)

    @_base.shape_check
    def __call__(self, graph: jraph.GraphsTuple) -> GradOut:
        """Compute the forward-mode Jacobian and write it into the graph."""
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
