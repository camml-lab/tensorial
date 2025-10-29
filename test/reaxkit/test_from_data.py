from jax import random
import jax.numpy as jnp
import omegaconf
import reax

from tensorial import reaxkit as rkit


def test_from_data(rng_key, test_trainer):
    # Now create the version as it would be in a configuration file
    cfg = omegaconf.DictConfig(
        {
            "avg": {"_target_": "reax.metrics.Average"},
            "min": {"_target_": "reax.metrics.Min"},
            "max": {"_target_": "reax.metrics.Max"},
            "std": {"_target_": "reax.metrics.Std"},
        }
    )

    batch_size = 9
    values = random.uniform(rng_key, (40,))
    loader = reax.data.ArrayLoader(values, batch_size=batch_size)

    # Use a trainer to run the stage
    from_data = rkit.FromData(cfg, test_trainer.engine, rngs=test_trainer.rngs, dataloader=loader)
    # This will do an in-place update of cfg
    test_trainer.run(from_data)

    results = cfg
    assert jnp.isclose(results["avg"], values.mean())
    assert jnp.isclose(results["min"], values.min())
    assert jnp.isclose(results["max"], values.max())
    assert jnp.isclose(results["std"], values.flatten().std())
