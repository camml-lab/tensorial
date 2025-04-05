import types

import jax
import jax.numpy as jnp
import numpy as np


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
