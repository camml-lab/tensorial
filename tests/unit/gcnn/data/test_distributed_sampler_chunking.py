"""End-to-end regression test for the DDP padding-overrun fix.

reax is framework-agnostic and cannot carry this test (it imports ``jraph``
and ``tensorial.gcnn``); the pure sampler-partition tests therefore live over
there (``reax/tests/data/test_samplers.py``).  This file covers the graph
side: it exercises reax's ``DistributedSampler`` (``shuffle=False``) together
with ``tensorial.gcnn.data._batching.GraphBatcher.calculate_padding``, which
computes a *per-batch* padding bound over contiguous batches of size
``batch_size``.  That bound is only valid if every per-rank batch is a
contiguous window of the dataset in its natural (post-split) order: under
PyTorch-style strided per-rank subsampling a rank's batch can land two "large"
graphs together and overflow the computed bound with
``RuntimeError: Given graph is too large for the given padding``.
"""

import jraph
import numpy as np
from reax.data import samplers

from tensorial import gcnn


def _square_graph(n_edge: int) -> jraph.GraphsTuple:
    """Build a trivial 2-node graph with ``n_edge`` edges (all self-loops)."""
    return jraph.GraphsTuple(
        n_node=np.array([2]),
        n_edge=np.array([n_edge]),
        nodes={"features": np.zeros((2, 1), dtype=np.float32)},
        edges={"features": np.zeros((n_edge, 1), dtype=np.float32)},
        globals={},
        senders=np.zeros(n_edge, dtype=np.int32),
        receivers=np.zeros(n_edge, dtype=np.int32),
    )


def test_ddp_no_shuffle_partition_respects_padding_bound():
    """Every per-rank batch must fit within the ``with_shuffle=False`` padding
    bound computed over the full dataset.

    Dataset edge counts: [5, 50, 5, 50], ``batch_size=2``, 2 ranks.
    - ``calculate_padding`` (``with_shuffle=False``) bounds a batch by the sum
      of the ``batch_size`` *largest* graphs (the only bound safe for DDP rank
      windows, which can contain a repeated index and need not be 0-aligned
      with the split's chunk boundaries): here top-2 == 50 + 50 == 100.
    - Chunked partition (rank 0 -> [0,1]; rank 1 -> [2,3]) gives [5, 50]
      for both ranks => 55 <= 100 bound, fits.
    """
    batch_size = 2
    edge_counts = np.array([5, 50, 5, 50], dtype=np.int64)

    graphs = [_square_graph(int(n)) for n in edge_counts]

    batcher = gcnn.data._batching.GraphBatcher(
        graphs, batch_size=batch_size, shuffle=False, pad=True
    )
    bound = int(batcher.padding.n_edges)
    # The bound is the sum of the `batch_size` largest graphs.
    assert bound == 100, f"expected bound 100, got {bound}"

    num_ranks = 2
    for rank in range(num_ranks):
        sampler = samplers.DistributedSampler(
            graphs,
            num_replicas=num_ranks,
            process_index=rank,
            shuffle=False,
            drop_last=False,
        )
        indices = list(sampler)
        assert len(indices) == batch_size

        # 1) Numerical invariant: batch's total edge count stays within the bound.
        batch_edges = int(sum(edge_counts[i] for i in indices))
        assert batch_edges <= bound, (
            f"rank {rank}: batch {indices} has {batch_edges} edges, " f"exceeds bound {bound}"
        )

        # 2) End-to-end: actually pulling a padded batch through the
        #    ``GraphBatcher`` must not overflow its padding.
        batcher.fetch(indices)  # raises RuntimeError on overflow
