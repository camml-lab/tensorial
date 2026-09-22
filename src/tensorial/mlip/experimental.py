from collections.abc import Callable

import jax
from jax.scipy import optimize as jopt
import jraph

from .. import as_array, gcnn


def minimize_fn(
    fun: gcnn.GraphFunction, what: gcnn.TreePathLike, wrt: gcnn.TreePathLike
) -> Callable:
    graph_fn = gcnn.adapt(fun, wrt, outs=(what,))

    def minim(
        graph: jraph.GraphsTuple, x0: jax.Array, *, method, tol=None, options=None
    ) -> jopt.OptimizeResults:
        # optimize() only takes 1D arrays, so flatten and un-flatten
        def to_minimize(value):
            value = value.reshape(x0.shape)
            return as_array(graph_fn(graph, value)).flatten()[0]

        res = jopt.minimize(to_minimize, x0.flatten(), method=method, tol=tol, options=options)
        res = res._replace(x=res.x.reshape(x0.shape), jac=res.jac.reshape(x0.shape))

        return res

    return minim
