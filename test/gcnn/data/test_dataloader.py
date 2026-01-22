import copy
import math

import jraph
import pytest

from tensorial import gcnn


@pytest.mark.parametrize(
    "batch_mode, expected_shape",
    [
        (gcnn.data.BatchMode.EXPLICIT, lambda bs: (bs, 2)),
        (gcnn.data.BatchMode.IMPLICIT, lambda bs: (bs + 1,)),
    ],
)
# In case of explicit batching, we expect batch.n_node.shape == (batch_size, 2)
# In case of implicit batching, we expect batch.n_node.shape == (batch_size + 1,)
def test_training_dataloader(cube_graph: jraph.GraphsTuple, batch_mode, expected_shape):

    dataset_size = 100
    batch_size = 7

    dset = [cube_graph for _ in range(dataset_size)]
    train_val_test_split = (0.8, 0.1, 0.1)

    num_batches = math.ceil(dataset_size * train_val_test_split[0] / batch_size)

    module_data = gcnn.data.GraphDataModule(
        dset,
        train_val_test_split=train_val_test_split,
        batch_size=batch_size,
        batch_mode=batch_mode,
    )

    module_data.setup(None)
    train_dl = module_data.train_dataloader()
    batches = tuple(train_dl)

    assert num_batches == len(batches)

    assert [batches[i][0].n_node.shape for i in range(num_batches)] == [
        expected_shape(batch_size) for _ in range(num_batches)
    ]
