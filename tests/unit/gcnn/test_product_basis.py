import functools

import e3nn_jax as e3j
import jax

from tensorial.gcnn import _product_basis


def test_symmetric_contraction():
    num_types = 4
    x = e3j.normal("0e + 0o + 1o + 1e + 2e + 2o", jax.random.PRNGKey(0), (32, 128))
    types = jax.random.randint(jax.random.PRNGKey(1), (32,), minval=0, maxval=num_types)

    contraction = _product_basis.SymmetricContraction(3, ["0e", "1o", "2e"], num_types=num_types)
    params = contraction.init(jax.random.PRNGKey(0), x, types)

    e3j.utils.assert_equivariant(
        functools.partial(contraction.apply, params, input_type=types), jax.random.PRNGKey(3), x
    )
