"""Module for functions performing expansion of functions with a basis."""

import functools

import jax.numpy as jnp

from . import bases, functions


@functools.singledispatch
def expand(  # pylint: disable=unused-argument
    basis: bases.RadialSphericalBasis, function: functions.Function
) -> jnp.array:
    """Expand a function in the given basis.

    This is a singledispatch function: the implementation is chosen based on
    the concrete types of ``basis`` and ``function``.
    """


@expand.register
def expand_(basis: bases.SimpleRadialSphericalBasis, function: functions.Function) -> jnp.array:
    """Expand a function in a :class:`.SimpleRadialSphericalBasis`."""
    if isinstance(function, functions.DiracDelta):
        return function.weight * basis.evaluate(function.pos)

    if isinstance(function, functions.Sum):
        return jnp.sum(expand(basis, function) for function in function.functions)

    raise TypeError(f"Unsupported function {function.__class__.__name__}")
