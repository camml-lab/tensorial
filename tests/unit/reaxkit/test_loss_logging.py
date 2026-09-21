"""The loss is logged as a raw value, so `reax` has to be told how to combine the per-batch losses
into an epoch value: averages are averaged, totals are added.
"""

from flax import linen
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import reax

from tensorial import reaxkit

# Ragged on purpose: 10 samples in batches of 4 gives batches of 4, 4 and 2, so weighting by batch
# size and weighting by batch are not the same thing
NUM_SAMPLES = 10
BATCH_SIZE = 4


class Squared(linen.Module):
    """A model with no parameters, so the loss is the same every epoch"""

    @linen.compact
    def __call__(self, x):
        return x**2


def mean_loss(predictions, targets):
    return optax.l2_loss(predictions, targets).mean()


def sum_loss(predictions, targets):
    return optax.l2_loss(predictions, targets).sum()


@pytest.fixture(name="dataset")
def dataset_fixture() -> np.ndarray:
    return np.random.default_rng(0).random((NUM_SAMPLES, 3))


def batches(dataset: np.ndarray) -> list[np.ndarray]:
    return [dataset[i : i + BATCH_SIZE] for i in range(0, len(dataset), BATCH_SIZE)]


def fit(dataset: np.ndarray, loss_fn, tmp_path, **kwargs) -> dict:
    module = reaxkit.ReaxModule(
        Squared(), loss_fn=loss_fn, optimizer=optax.adamw(learning_rate=0.01), output=None, **kwargs
    )
    module.debug = False  # `Squared` has no parameters, so there are no gradient norms to log
    trainer = reax.Trainer(default_root_dir=tmp_path)
    # Inputs and targets are the same array: the loss compares the model output with its input
    loader = reax.data.ArrayLoader((dataset, dataset), batch_size=BATCH_SIZE)
    trainer.fit(module, train_dataloaders=loader, max_epochs=1)

    return trainer.logged_metrics


def test_mean_loss_is_averaged_over_samples(dataset, tmp_path):
    """A loss that averages over its batch stays on that scale, weighted by the samples behind it,
    so the short final batch doesn't count as a whole one"""
    per_batch = [float(mean_loss(batch**2, batch)) for batch in batches(dataset)]
    weights = [len(batch) for batch in batches(dataset)]

    logged = fit(dataset, mean_loss, tmp_path, loss_reduction="mean")["train/loss"]

    assert jnp.isclose(logged, np.average(per_batch, weights=weights), rtol=1e-5)
    # The bug this guards against: the mean divided by the batch size a second time
    assert not jnp.isclose(logged, sum(per_batch) / NUM_SAMPLES)


def test_sum_loss_is_totalled_over_batches(dataset, tmp_path):
    """A loss returning a total per batch gives the total over the epoch"""
    per_batch = [float(sum_loss(batch**2, batch)) for batch in batches(dataset)]

    logged = fit(dataset, sum_loss, tmp_path, loss_reduction="sum")["train/loss"]

    assert jnp.isclose(logged, sum(per_batch), rtol=1e-5)


def test_plain_callable_defaults_to_mean(dataset, tmp_path):
    """A loss function that can't be asked how it reduces is assumed to return an average"""
    assert (
        fit(dataset, mean_loss, tmp_path)["train/loss"]
        == fit(dataset, mean_loss, tmp_path, loss_reduction="mean")["train/loss"]
    )


def test_reduction_is_taken_from_the_loss_function(dataset, tmp_path):
    """A loss that says how it reduces doesn't need to be told twice"""

    class MeanLoss:
        reduction = "mean"

        def __call__(self, predictions, targets):
            return mean_loss(predictions, targets)

    assert (
        fit(dataset, MeanLoss(), tmp_path)["train/loss"]
        == fit(dataset, mean_loss, tmp_path, loss_reduction="mean")["train/loss"]
    )
