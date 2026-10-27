from jax import random
import jax.numpy as jnp
import omegaconf
import pytest
import reax

from tensorial import reaxkit as rkit


class _CountingDataModule(reax.DataModule):
    """A data module that records how often its data lifecycle hooks are called."""

    def __init__(self, values, batch_size):
        super().__init__()
        self.values = values
        self.batch_size = batch_size
        self.prepare_calls = 0
        self.setup_calls = 0
        self.teardown_calls = 0

    def prepare_data(self):
        self.prepare_calls += 1

    def setup(self, engine, /, *, stage=None):
        self.setup_calls += 1

    def teardown(self, engine, /, *, stage=None):
        self.teardown_calls += 1

    def train_dataloader(self):
        return reax.data.ArrayLoader(self.values, batch_size=self.batch_size)


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
    from_data = rkit.FromData(cfg, test_trainer.engine, dataloader=loader)
    # This will do an in-place update of cfg
    test_trainer.run(from_data)

    results = cfg
    assert jnp.isclose(results["avg"], values.mean())
    assert jnp.isclose(results["min"], values.min())
    assert jnp.isclose(results["max"], values.max())
    assert jnp.isclose(results["std"], values.flatten().std())


def test_from_data_runs_data_lifecycle_once(rng_key, test_trainer):
    # `FromData` must not nest a stage that repeats the data lifecycle on the shared
    # data manager.  The stage's own hooks should run prepare/setup/teardown exactly once.
    cfg = omegaconf.DictConfig(
        {
            "avg": {"_target_": "reax.metrics.Average"},
            "min": {"_target_": "reax.metrics.Min"},
            "max": {"_target_": "reax.metrics.Max"},
            "std": {"_target_": "reax.metrics.Std"},
        }
    )

    values = random.uniform(rng_key, (40,))
    datamodule = _CountingDataModule(values, batch_size=9)

    from_data = rkit.FromData(cfg, test_trainer.engine, datamodule=datamodule)
    test_trainer.run(from_data)

    assert datamodule.prepare_calls == 1
    assert datamodule.setup_calls == 1
    assert datamodule.teardown_calls == 1

    assert jnp.isclose(cfg["avg"], values.mean())
    assert jnp.isclose(cfg["min"], values.min())
    assert jnp.isclose(cfg["max"], values.max())
    assert jnp.isclose(cfg["std"], values.flatten().std())


@pytest.mark.parametrize("ignore_missing", [True, False])
def test_from_data_propagates_ignore_missing(ignore_missing, monkeypatch, rng_key, test_trainer):
    seen = []
    real_stats_epoch = reax.stages.StatsEpoch

    class SpyStatsEpoch(real_stats_epoch):
        def __init__(self, stage, **kwargs):
            seen.append(kwargs.get("ignore_missing"))
            super().__init__(stage, **kwargs)

    monkeypatch.setattr(reax.stages, "StatsEpoch", SpyStatsEpoch)

    cfg = omegaconf.DictConfig({"avg": {"_target_": "reax.metrics.Average"}})
    values = random.uniform(rng_key, (40,))
    loader = reax.data.ArrayLoader(values, batch_size=9)

    from_data = rkit.FromData(
        cfg,
        test_trainer.engine,
        dataloader=loader,
        ignore_missing=ignore_missing,
    )
    test_trainer.run(from_data)

    assert seen == [ignore_missing]
    assert jnp.isclose(cfg["avg"], values.mean())
