import jax.numpy as jnp
import numpy as np
import omegaconf

from tensorial import config, data, metrics


def test_calculate_stats():
    n_points = 100
    array = np.random.rand(n_points, 2)  # Some fake inputs
    dataset = data.ArrayLoader(array, batch_size=32)

    # Let's create a fake configuration with some stats we want to calculate
    stats = {stat: f"data_{stat}" for stat in ("min", "max", "mean")}
    from_data = omegaconf.DictConfig(stats)

    res = config.calculate_stats(from_data, dataset)
    # Just check it doesn't return anything, should update in place
    assert res is None

    # Check that our configuration has been updated with the correct metric values
    for name, value in from_data.items():
        res = metrics.get(name).create(array).compute()
        assert jnp.allclose(res, value)
