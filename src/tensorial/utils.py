"""Internal helpers: backend inference, zero/one helpers, optional imports."""

import importlib
import types

import e3nn_jax as e3j
import jax
import jax.numpy as jnp
import numpy as np

from tensorial.typing import IntoIrreps


def infer_backend(pytree) -> types.ModuleType:
    """Infer the array backend (``numpy`` or ``jax.numpy``) used by a pytree.

    Args:
        pytree: a pytree whose leaves are arrays.

    Returns:
        the numpy or jax.numpy module to use for subsequent array ops.

    Raises:
        ValueError: if the pytree mixes numpy and JAX arrays.
    """
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
    irreps: IntoIrreps, leading_shape: tuple = (), dtype: jnp.dtype = None, np_=jnp
) -> e3j.IrrepsArray:
    """Create a zero-valued :class:`e3nn_jax.IrrepsArray` with the given irreps.

    Args:
        irreps: the irreps of the resulting array.
        leading_shape: leading batch shape to prepend to the irrep axis.
        dtype: optional dtype of the resulting array.
        np_: which backend (``numpy`` or ``jax.numpy``) to use for ``zeros``.

    Returns:
        the zero-valued irreps array.
    """
    irreps = e3j.Irreps(irreps)
    array = np_.zeros(leading_shape + (irreps.dim,), dtype=dtype)
    return e3j.IrrepsArray(irreps, array, zero_flags=(True,) * len(irreps))


def zeros_like(irreps_array: e3j.IrrepsArray) -> e3j.IrrepsArray:
    """Return a zero-valued :class:`e3nn_jax.IrrepsArray` with the same shape/irreps as the input.

    Args:
        irreps_array: the irreps array to mimic.

    Returns:
        the zero-valued irreps array.
    """
    np_ = infer_backend(irreps_array.array)
    return zeros(irreps_array.irreps, irreps_array.shape[:-1], irreps_array.dtype, np_=np_)


def ones(
    irreps: IntoIrreps, leading_shape: tuple = (), dtype: jnp.dtype = None, np_=jnp
) -> e3j.IrrepsArray:
    """Create a one-valued :class:`e3nn_jax.IrrepsArray` with the given irreps.

    Args:
        irreps: the irreps of the resulting array.
        leading_shape: leading batch shape to prepend to the irrep axis.
        dtype: optional dtype of the resulting array.
        np_: which backend (``numpy`` or ``jax.numpy``) to use for ``ones``.

    Returns:
        the one-valued irreps array.
    """
    irreps = e3j.Irreps(irreps)
    array = np_.ones(leading_shape + (irreps.dim,), dtype=dtype)
    return e3j.IrrepsArray(irreps, array, zero_flags=(False,) * len(irreps))


def ones_like(irreps_array: e3j.IrrepsArray) -> e3j.IrrepsArray:
    """Return a one-valued :class:`e3nn_jax.IrrepsArray` with the same shape/irreps as the input.

    Args:
        irreps_array: the irreps array to mimic.

    Returns:
        the one-valued irreps array.
    """
    np_ = infer_backend(irreps_array.array)
    return ones(irreps_array.irreps, irreps_array.shape[:-1], irreps_array.dtype, np_=np_)


def optional_import(name: str, extra: str | None = None):
    """Import a module and raise a helpful error if it is missing.

    Args:
        name: the dotted module path to import.
        extra: optional PEP 508 extra on the distribution, used in the install hint.

    Returns:
        the imported module.

    Raises:
        ImportError: if the module is not installed, including a hint on how to install it.
    """
    try:
        return importlib.import_module(name)
    except ImportError as e:
        if extra:
            hint = f" Install it with `pip install mylib[{extra}]`."
        else:
            hint = f" Install it with `pip install {name.split('.')[0]}`."

        raise ImportError(
            f"'{name}' is required for this feature but is not installed.{hint}"
        ) from e
