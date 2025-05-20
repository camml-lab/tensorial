import jax

import tensorial


def create_test_validate(
    rng_key,
    n_train: int,
    n_validate: int,
    samples_per_batch=10,
    features=4,
) -> tuple[tensorial.data.ArrayLoader, tensorial.data.ArrayLoader]:
    keys = jax.random.split(rng_key, 5)

    train = tensorial.data.ArrayLoader(
        jax.random.uniform(keys[0], (n_train, samples_per_batch, features)),
        jax.random.uniform(keys[1], (n_train, samples_per_batch, features)),
    )
    validate = tensorial.data.ArrayLoader(
        jax.random.uniform(keys[2], (n_validate, samples_per_batch, features)),
        jax.random.uniform(keys[3], (n_validate, samples_per_batch, features)),
    )
    return train, validate
