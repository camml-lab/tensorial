# -*- coding: utf-8 -*-
from collections.abc import Sequence
import functools
from typing import Any, Union

from flax import linen
import jax
import jax.numpy as jnp
import jraph
from pytray import tree

import tensorial

from . import _base, utils

__all__ = ("Grad",)

TreePath = tuple[Any, ...]


class Grad(linen.Module):
    func: _base.GraphFunction
    of: str
    wrt: Union[str, TreePath]
    out_field: Union[str, Sequence[str]] = None
    sign: float = 1.0

    def setup(self):
        # pylint: disable=attribute-defined-outside-init
        if isinstance(self.wrt, str):
            self._wrt = (utils.path_from_str(self.wrt),)
        elif isinstance(self.wrt, (list, tuple)):
            self._wrt = tuple(map(utils.path_from_str(self.wrt)))
        else:
            raise ValueError(
                f"wrt must be str or list or tuple thereof, got {type(self.wrt).__name__}"
            )
        self._of = utils.path_from_str(self.of)

        if self.out_field is None:
            derivs = []
            for wrt in self._wrt:
                derivs.append(self._of[:-1] + (f"d{'.'.join(self._of[1:])}/d{wrt[-1]}",))
            self._out_field = tuple(derivs)
        else:
            self._out_field = [utils.path_from_str(self.out_field)]

        self._grad_fn = jax.grad(grad_shim, argnums=4, has_aux=True)

    def __call__(self, graph: jraph.GraphsTuple):
        graph_dict = graph._asdict()
        wrt_variables = tuple(map(lambda path: tree.get_by_path(graph_dict, path), self._wrt))

        grads, graph_out = self._grad_fn(self.func, graph, self._of, self._wrt, *wrt_variables)
        grads = (grads,)

        # Add the gradient quantity to the output graph
        out_graph_dict = graph_out._asdict()
        for out_path, value in zip(self._out_field, grads):
            tree.set_by_path(out_graph_dict, out_path, self.sign * value)
        return jraph.GraphsTuple(**out_graph_dict)


def grad_shim(
    fn: _base.GraphFunction,
    graph: jraph.GraphsTuple,
    of: tuple,
    paths: tuple[TreePath],
    *wrt_variables,
) -> tuple[jax.Array, jraph.GraphsTuple]:
    def repl(path, val):
        try:
            idx = paths.index(tuple(map(key_to_str, path)))
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


@functools.singledispatch
def key_to_str(key) -> str:
    raise ValueError(key)


@key_to_str.register
def attr_key_to_str(key: jax.tree_util.GetAttrKey):
    return key.name


@key_to_str.register
def dict_key_to_str(key: jax.tree_util.DictKey):
    return str(key.key)


@key_to_str.register
def sequence_key_to_str(key: jax.tree_util.SequenceKey):
    return str(key.idx)
