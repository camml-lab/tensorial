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
