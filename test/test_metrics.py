from jax import random
import jax.numpy as jnp
import optax

from tensorial import metrics


def test_aggregation(rng_key):
    values = random.uniform(rng_key, (100,))
    avg = metrics.Average()
    assert jnp.allclose(avg.update(values).compute(), values.mean())


def test_mean_square_error(rng_key):
    rng_key, *keys = random.split(rng_key, 3)
    n_batches = 4
    values = random.uniform(keys[0], (n_batches, 10))
    targets = random.uniform(keys[1], (n_batches, 10))

    mse = metrics.MeanSquaredError()
    for prediction, target in zip(values, targets):
        mse = mse.update(prediction, target)

    assert jnp.isclose(mse.compute(), optax.squared_error(values, targets).mean())
    # Check the convenience function gives us the right type
    assert metrics.get("mse") is metrics.MeanSquaredError


def test_root_mean_square_error(rng_key):
    rng_key, *keys = random.split(rng_key, 3)
    n_batches = 4
    predictions = random.uniform(keys[0], (n_batches, 10))
    targets = random.uniform(keys[1], (n_batches, 10))

    mse = metrics.RootMeanSquareError()
    for prediction, target in zip(predictions, targets):
        mse = mse.update(prediction, target)

    assert jnp.isclose(mse.compute(), jnp.sqrt(optax.squared_error(predictions, targets).mean()))
    # Check the convenience function gives us the right type
    assert metrics.get("rmse") is metrics.RootMeanSquareError


def test_mae(rng_key):
    rng_key, *keys = random.split(rng_key, 3)
    n_batches = 4
    predictions = random.uniform(keys[0], (n_batches, 10))
    targets = random.uniform(keys[1], (n_batches, 10))

    mse = metrics.MeanAbsoluteError()
    for prediction, target in zip(predictions, targets):
        mse = mse.update(prediction, target)

    assert jnp.isclose(mse.compute(), jnp.abs(predictions - targets).mean())
    # Check the convenience function gives us the right type
    assert metrics.get("mae") is metrics.MeanAbsoluteError


def test_from_fn(rng_key):
    n_batches = 4
    values = random.uniform(rng_key, (n_batches, 10))

    # Let's create a fake function, where we calculate the mean of the squares
    metric = metrics.from_fun(lambda values: metrics.Average.create(values**2)).empty()
    mean = jnp.mean(values**2)

    for batch in values:
        metric = metric.update(batch)

    assert jnp.isclose(metric.compute(), mean)
