# -*- coding: utf-8 -*-
from typing import Callable

import jax
import jax.nn

ActivationFunction = Callable[[jax.Array], jax.Array]


def get_jaxnn_activation(name: str) -> ActivationFunction:
    """
    Returns the activation function with `name` form the jax.nn module

    :param name: the name of the function (as used in `jax.nn`)
    :return: the activation function
    """
    try:
        return getattr(jax.nn, name)
    except AttributeError:
        raise ValueError(f"Activation function '{name}' not found in jax.nn") from None


def prepare_mask(mask: jax.Array, array: jax.Array) -> jax.Array:
    """
    Prepare a mask for use with jnp.where(mask, array, ...).  This needs to be done to make sure the
    mask is of the right shape to be compatible with such an operation.  The other alternative is

        ``jnp.where(mask, array.T, ...).T``

    but this sometimes leads to creating a copy when doing one or both of the transposes.  I'm not
    sure why, but this approach seems to avoid the problem.

    :param mask: the mask to prepare
    :param array: the array the mask will be applied to
    :return: the prepared mask, typically this is just padded with extra dimensions (or reduced)
    """
    return mask.reshape(-1, *(1,) * len(array.shape[1:]))
