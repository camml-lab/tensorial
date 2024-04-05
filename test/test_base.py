# -*- coding: utf-8 -*-
import e3nn_jax as e3j
import jax
import jax.numpy as jnp
import pytest

import tensorial


class Particle(tensorial.IrrepsObj):
    """Class for a particle with some attributes"""

    pos = e3j.Irreps('1e')
    mass = tensorial.Attr('0e')
    shielding = tensorial.CartesianTensor('ij=ji', i='1e')


def test_basics():
    pos = jnp.array([5.2, 5.6, 1.1])
    mass = 12.45
    shielding = jnp.array([[1.2, 0.0, 1.1], [0.0, 0.0, 1.4], [1.1, 1.4, 3.1]])

    # Make sure we get an error if we don't pass all the attributes
    with pytest.raises(KeyError):
        tensorial.create_tensor(Particle, dict(pos=pos))

    # Test creating a tensor and extracting values
    tensor = tensorial.create_tensor(Particle, dict(pos=pos, mass=mass, shielding=shielding))
    assert jnp.allclose(tensorial.get(Particle, tensor, 'pos').array, pos)
    assert jnp.allclose(tensorial.get(Particle, tensor, 'mass').array[0].item(), mass)
    # assert jnp.allclose(tensorial.get(Particle, tensor, "shielding"), shielding)

    # Test getting whole tensor
    assert tensorial.get(Particle, tensor) is tensor


def test_cartesian_tensor():
    key = jax.random.PRNGKey(0)
    irreps = e3j.Irreps('1o')
    cart = tensorial.CartesianTensor('ij=ji', i=irreps)

    # Let's create a random Cartesian tensor
    cart_tensor = jax.random.uniform(key, (irreps.dim, irreps.dim))
    cart_tensor = cart_tensor @ cart_tensor.T  # symmetrise

    # Make sure the round trip conversion leads back to the original tensor
    result = cart.from_tensor(cart.create_tensor(cart_tensor))
    jnp.allclose(cart_tensor, result)
