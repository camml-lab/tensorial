import jax.numpy as jnp
import pytest
from reax.data.utils import extract_batch_size
from reax.results import ResultCollection

from tensorial import gcnn


@pytest.mark.parametrize("batch_mode", [gcnn.data.BatchMode.IMPLICIT, gcnn.data.BatchMode.EXPLICIT])
def test_cumulated_batch_size_effect(cube_graph, batch_mode):
    """Test that the cumulated batch size correctly influences the computed metric
    """
    dataset_size = 9
    dset = [cube_graph for _ in range(dataset_size)]

    B = 5
    dm = gcnn.data.GraphDataModule(
        dset,
        train_val_test_split=(1.0, 0.0, 0.0),
        batch_size=B,
        batch_mode=batch_mode,
    )
    dm.setup(None)

    loader = dm.train_dataloader()
    batches = tuple(loader)
    batch1 = batches[0]
    batch2 = batches[1]

    bs1 = extract_batch_size(batch1)
    bs2 = extract_batch_size(batch2)

    mask1 = batch1[0].globals["mask"]
    mask2 = batch2[0].globals["mask"]

    B1 = int(mask1.sum())
    B2 = int(mask2.sum())

    # Verify that batching actually results in the correct extracted batch size
    assert bs1 == B1, f"Extracted batch size 1 was {bs1}, expected {B1}"
    assert bs2 == B2, f"Extracted batch size 2 was {bs2}, expected {B2}"

    metrics = ResultCollection()

    # The loss is typically a sum over the batch
    loss_sum1 = 15.0 * B1
    loss_sum2 = 10.0 * B2

    # Log metrics as the trainer would
    metrics.log("train", "loss", loss_sum1, batch_idx=0, on_epoch=True, batch_size=bs1)
    metrics.log("train", "loss", loss_sum2, batch_idx=1, on_epoch=True, batch_size=bs2)

    # Calculate metric outcome using ArrayResultMetric compute
    metric = metrics["train.loss"].metric
    computed = metric.compute()

    # Compute statistically correct expected mean: (Sum over all valid graphs) / (Total Number of graphs)
    expected_mean = (loss_sum1 + loss_sum2) / (B1 + B2)
    assert jnp.isclose(
        computed, expected_mean
    ), f"Expected batch mean {expected_mean}, but got {computed}. Cumulated batch size calculation is incorrect!"
