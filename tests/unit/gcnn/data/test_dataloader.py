import copy
import math

import jraph
import numpy as np
import pytest
from reax.data import samplers

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


def _trivial_graphs(n: int) -> list[jraph.GraphsTuple]:
    """A list of identical, non-empty graphs to seed a ``GraphLoader``."""
    graph = jraph.GraphsTuple(
        n_node=np.array([2]),
        n_edge=np.array([3]),
        nodes={"f": np.ones((2, 1), dtype=np.float32)},
        edges={"f": np.ones((3, 1), dtype=np.float32)},
        globals={"g": np.ones((1,), dtype=np.float32)},
        senders=np.zeros(3, dtype=np.int32),
        receivers=np.zeros(3, dtype=np.int32),
    )
    return [graph for _ in range(n)]


@pytest.mark.parametrize(
    "batch_mode",
    [
        gcnn.data.BatchMode.EXPLICIT,  # n_node has shape (batch_size, 2)
        gcnn.data.BatchMode.IMPLICIT,  # n_node has shape (batch_size + 1,)
    ],
)
def test_empty_batch_yields_fully_masked_graph(batch_mode):
    """The distributed sampler may hand a rank a completely empty batch (``[]``) to
    keep the per-rank step count uniform.  ``GraphLoader`` must turn that into a valid
    graph that is *entirely* masked out -- the same shape a padded batch would have, but
    with every node, edge and graph flagged as padding.
    """
    batch_size = 4
    graphs = _trivial_graphs(2)

    # A rank that owns no real global step: its sampler yields a single empty batch.
    inner = samplers.BatchSampler(
        samplers.SequentialSampler(len(graphs)), batch_size=batch_size, drop_last=False
    )
    empty_sampler = samplers.DistributedSampler(inner=inner, num_replicas=2, process_index=1)
    assert list(empty_sampler) == [[]], "test setup assumes rank 1 yields one empty batch"

    loader = gcnn.data.GraphLoader(
        graphs,
        batch_size=batch_size,
        pad=True,
        batch_mode=batch_mode,
        sampler=empty_sampler,
    )

    batches = tuple(loader)
    assert len(batches) == 1
    batch = batches[0][0]

    # Static shape: the empty batch is still fully padded to the batch size.
    if batch_mode is gcnn.data.BatchMode.EXPLICIT:
        assert batch.n_node.shape == (batch_size, 2)
    else:  # IMPLICIT
        assert batch.n_node.shape == (loader.padding.n_graphs,)

    # All real graphs are empty.
    assert int(np.asarray(batch.n_node).sum()) == 0
    assert int(np.asarray(batch.n_edge).sum()) == 0

    # Every node, edge and graph is masked out (fully masked graph).
    node_mask = np.asarray(batch.nodes[gcnn.keys.MASK])
    edge_mask = np.asarray(batch.edges[gcnn.keys.MASK])
    graph_mask = np.asarray(batch.globals[gcnn.keys.MASK])
    assert not node_mask.any(), "no node should be marked as real in an empty batch"
    assert not edge_mask.any(), "no edge should be marked as real in an empty batch"
    assert not graph_mask.any(), "no graph should be marked as real in an empty batch"
