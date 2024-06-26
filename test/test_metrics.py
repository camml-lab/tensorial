from jax import random
import jax.numpy as jnp
import optax
import pytest

from tensorial import metrics


def test_mean_square_error(rng_key):
    rng_key, *keys = random.split(rng_key, 3)
    n_batches = 4
    values = random.uniform(keys[0], (n_batches, 10))
    targets = random.uniform(keys[1], (n_batches, 10))

    mse = metrics.MeanSquaredError.empty()
    for prediction, target in zip(values, targets):
        mse = mse.merge(mse.from_model_output(prediction, target))

    assert jnp.isclose(mse.compute(), optax.squared_error(values, targets).mean())
    # Check the convenience function gives us the right type
    assert metrics.metric("mse") is metrics.MeanSquaredError


def test_root_mean_square_error(rng_key):
    rng_key, *keys = random.split(rng_key, 3)
    n_batches = 4
    predictions = random.uniform(keys[0], (n_batches, 10))
    targets = random.uniform(keys[1], (n_batches, 10))

    mse = metrics.RootMeanSquareError.empty()
    for prediction, target in zip(predictions, targets):
        mse = mse.merge(mse.from_model_output(prediction, target))

    assert jnp.isclose(mse.compute(), jnp.sqrt(optax.squared_error(predictions, targets).mean()))
    # Check the convenience function gives us the right type
    assert metrics.metric("rmse") is metrics.RootMeanSquareError


def test_mae(rng_key):
    rng_key, *keys = random.split(rng_key, 3)
    n_batches = 4
    predictions = random.uniform(keys[0], (n_batches, 10))
    targets = random.uniform(keys[1], (n_batches, 10))

    mse = metrics.MeanAbsoluteError.empty()
    for prediction, target in zip(predictions, targets):
        mse = mse.merge(mse.from_model_output(prediction, target))

    assert jnp.isclose(mse.compute(), jnp.abs(predictions - targets).mean())
    # Check the convenience function gives us the right type
    assert metrics.metric("mae") is metrics.MeanAbsoluteError


def test_simple_metrics(rng_key):
    vals = jnp.arange(10)

    # This class needs to be used with the ``create()`` method and so it should not be able to
    # construct it otherwise
    with pytest.raises(NotImplementedError):
        metrics.StatMetric.from_model_output(vals)

    min_metric = metrics.StatMetric.create(jnp.min)

    metric = min_metric.from_model_output(vals)
    assert metric.compute() == jnp.min(vals)

    metric = min_metric.empty().merge(min_metric.from_model_output(vals))
    assert metric.compute() == jnp.min(vals)
