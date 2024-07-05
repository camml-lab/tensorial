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


def test_gaussian(rng_key):
    pos = jax.random.uniform(rng_key, shape=(3,))
    weight = jax.random.uniform(rng_key)
    sigma = jax.random.uniform(rng_key)

    gaussian = signals.functions.IsotropicGaussian(pos, sigma, weight=weight)

    gaussian(pos)
    # final = weight / (sigma * jnp.sqrt(2 * jnp.pi))
    # assert val == weight / (sigma * jnp.sqrt(2 * jnp.pi))
