# -*- coding: utf-8 -*-
from jax import random
import jax.numpy as jnp
import optax

from tensorial import metrics


def test_mean_square_error(rng_key):
    rng_key, *keys = random.split(rng_key, 3)
    n_batches = 4
    predictions = random.uniform(keys[0], (n_batches, 10))
    targets = random.uniform(keys[1], (n_batches, 10))

    mse = metrics.MeanSquaredError.empty()
    for prediction, target in zip(predictions, targets):
        mse = mse.merge(mse.from_model_output(prediction, target))

    assert jnp.isclose(mse.compute(), optax.squared_error(predictions, targets).mean())


def test_root_mean_square_error(rng_key):
    rng_key, *keys = random.split(rng_key, 3)
    n_batches = 4
    predictions = random.uniform(keys[0], (n_batches, 10))
    targets = random.uniform(keys[1], (n_batches, 10))

    mse = metrics.RootMeanSquareError.empty()
    for prediction, target in zip(predictions, targets):
        mse = mse.merge(mse.from_model_output(prediction, target))

    assert jnp.isclose(mse.compute(), jnp.sqrt(optax.squared_error(predictions, targets).mean()))
