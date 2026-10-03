import jax.numpy as jnp
import pytest
from reax import results
from reax.data import utils

from tensorial import gcnn
from tensorial.data import PassthroughFetcher


@pytest.mark.parametrize("batch_mode", [gcnn.data.BatchMode.IMPLICIT, gcnn.data.BatchMode.EXPLICIT])
def test_batch_size_and_logged_loss(cube_graph, batch_mode):
    """Test that padded batches report the number of real graphs, and that per-batch losses are
    combined over batches"""
    dataset_size = 9
    dset = [cube_graph for _ in range(dataset_size)]

    batch_size = 5
    dm = gcnn.data.GraphDataModule.from_random_split(
        PassthroughFetcher(dset),
        train_val_test_split=(1.0, 0.0, 0.0),
        batch_size=batch_size,
        batch_mode=batch_mode,
    )
    dm.prepare_data()
    dm.setup(None)

    loader = dm.train_dataloader()
    batches = tuple(loader)
    batch1 = batches[0]
    batch2 = batches[1]

    bs1 = utils.extract_batch_size(batch1)
    bs2 = utils.extract_batch_size(batch2)

    mask1 = batch1[0].globals["mask"]
    mask2 = batch2[0].globals["mask"]

    batch_size1 = int(mask1.sum())
    batch_size2 = int(mask2.sum())

    # Verify that batching actually results in the correct extracted batch size
    assert bs1 == batch_size1, f"Extracted batch size 1 was {bs1}, expected {batch_size1}"
    assert bs2 == batch_size2, f"Extracted batch size 2 was {bs2}, expected {batch_size2}"

    metrics = results.ResultCollection()

    # The loss is a mean over the batch, as `gcnn.Loss` produces by default
    loss1 = 15.0
    loss2 = 10.0

    # Log metrics as the trainer would
    metrics.log("train", "loss", loss1, batch_idx=0, on_epoch=True, batch_size=bs1)
    metrics.log("train", "loss", loss2, batch_idx=1, on_epoch=True, batch_size=bs2)

    # Calculate metric outcome using ArrayResultMetric compute
    metric = metrics["train.loss"].metric
    computed = metric.compute()

    # Each per-batch mean is weighted by the number of real graphs behind it, so a padded final
    # batch counts for the graphs it actually holds rather than as a whole batch
    expected_mean = (loss1 * batch_size1 + loss2 * batch_size2) / (batch_size1 + batch_size2)
    assert jnp.isclose(computed, expected_mean), (
        f"Expected the mean over graphs {expected_mean}, but got {computed}"
    )
