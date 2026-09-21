import jax
import jax.random

from tensorial import signals


def test_delta(rng_key):
    pos = jax.random.uniform(rng_key, shape=(3,))
    weight = jax.random.uniform(rng_key)
    delta = signals.functions.DiracDelta(pos, weight)

    assert delta(pos) == weight
    assert delta(jax.random.uniform(rng_key)) == 0.0

    # vmapped = jax.vmap(delta)
    # print(vmapped(jax.random.uniform(key, shape=(10, 3))))


def test_gaussian_values(rng_key):
    import jax.numpy as jnp

    pos = jax.random.uniform(rng_key, shape=(3,))
    weight = 2.0
    sigma = 1.5

    gaussian = signals.functions.IsotropicGaussian(pos, sigma, weight=weight)

    # Peak at the centroid.
    expected_peak = weight / jnp.sqrt(2 * jnp.pi)
    assert gaussian(pos) == expected_peak

    # Falls off monotonically with distance.
    near = gaussian(pos + jnp.array([0.1, 0.0, 0.0]))
    far = gaussian(pos + jnp.array([2.0, 0.0, 0.0]))
    assert near < expected_peak
    assert far < near


def test_sum_flattens_nested_sums(rng_key):
    from tensorial.signals.functions import DiracDelta, Sum

    a = DiracDelta(jax.random.uniform(rng_key, shape=(3,)))
    b = DiracDelta(jax.random.uniform(rng_key))
    c = DiracDelta(jax.random.uniform(rng_key))

    flat = Sum((a, b))
    assert flat.functions == (a, b)

    nested = Sum((a, Sum((b, c))))
    assert nested.functions == (a, b, c)


def test_sum_add(rng_key):
    a = signals.functions.DiracDelta(jax.random.uniform(rng_key, shape=(3,)), weight=1.0)
    b = signals.functions.DiracDelta(jax.random.uniform(rng_key, shape=(3,)), weight=1.0)

    total = a + b
    assert isinstance(total, signals.functions.Sum)
    assert total.functions == (a, b)


def test_sum_evaluate_is_linear(rng_key):
    from tensorial.signals.functions import DiracDelta

    pos = jax.random.uniform(rng_key, shape=(3,))
    a = DiracDelta(pos, weight=3.0)
    b = DiracDelta(pos, weight=4.0)

    # Sum is built from ``a + b``.  The individual members evaluate linearly.
    summed = a + b
    total = sum(func(pos) for func in summed.functions)
    assert total == 7.0
