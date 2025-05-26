import types

import e3nn_jax as e3j
import jax
import jax.numpy as jnp
import numpy as np

from . import typing


def infer_backend(pytree) -> types.ModuleType:
    """Try to infer a backend from the passed pytree"""
    any_numpy = any(isinstance(x, np.ndarray) for x in jax.tree_util.tree_leaves(pytree))
    any_jax = any(isinstance(x, jax.Array) for x in jax.tree_util.tree_leaves(pytree))
    if any_numpy and any_jax:
        raise ValueError("Cannot mix numpy and jax arrays")

    if any_numpy:
        return np

    if any_jax:
        return jnp

    return jnp


def zeros(
    irreps: typing.IntoIrreps, leading_shape: tuple = (), dtype: jnp.dtype = None, np_=jnp
) -> e3j.IrrepsArray:
    r"""Create an IrrepsArray of zeros."""
    irreps = e3j.Irreps(irreps)
    array = np_.zeros(leading_shape + (irreps.dim,), dtype=dtype)
    return e3j.IrrepsArray(irreps, array, zero_flags=(True,) * len(irreps))


def zeros_like(irreps_array: e3j.IrrepsArray) -> e3j.IrrepsArray:
    r"""Create an IrrepsArray of zeros with the same shape as another IrrepsArray."""
    np_ = infer_backend(irreps_array.array)
    return zeros(irreps_array.irreps, irreps_array.shape[:-1], irreps_array.dtype, np_=np_)


def ones(
    irreps: typing.IntoIrreps, leading_shape: tuple = (), dtype: jnp.dtype = None, np_=jnp
) -> e3j.IrrepsArray:
    r"""Create an IrrepsArray of ones."""
    irreps = e3j.Irreps(irreps)
    array = np_.ones(leading_shape + (irreps.dim,), dtype=dtype)
    return e3j.IrrepsArray(irreps, array, zero_flags=(False,) * len(irreps))


def ones_like(irreps_array: e3j.IrrepsArray) -> e3j.IrrepsArray:
    r"""Create an IrrepsArray of ones with the same shape as another IrrepsArray."""
    np_ = infer_backend(irreps_array.array)
    return ones(irreps_array.irreps, irreps_array.shape[:-1], irreps_array.dtype, np_=np_)
