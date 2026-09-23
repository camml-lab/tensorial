"""Regression tests for the DDP padding-overrun bug.

Under DDP, reax's ``DistributedSampler`` (``shuffle=False``) hands each rank
a *contiguous block* of the dataset, starting at ``rank * num_samples`` (where
``num_samples`` is the number of items per rank).  The batches a rank then
produces (via reax's ``BatchSampler``) are therefore **windows into the
full split that are not 0-aligned with the full split's chunk boundaries**.

The existing ``GraphBatcher.calculate_padding`` (``with_shuffle=False``) takes
the max over *0-aligned* chunks of ``batch_size``.  A rank's first batch is a
sliding window that can start at a non-0-aligned offset, and it can exceed
the budget computed from the 0-aligned windows -- ``jraph.pad_with_graphs``
then fails with ``RuntimeError: Given graph is too large for the given
padding`` (signature: ``n_edge -k`` for some small positive ``k``).

These tests pin down that the padding budget must cover **all** windows of
size ``batch_size`` that DDP's per-rank block can produce -- not just the
0-aligned ones.
"""

import jax.numpy as jnp
import jraph

from tensorial import gcnn


def _make_graph(n_edges: int) -> jraph.GraphsTuple:
    """A single connected ``n_edges``-edge graph on ``n_edges + 1`` nodes."""
    node_count = n_edges + 1
    return jraph.GraphsTuple(
        n_node=jnp.array([node_count]),
        n_edge=jnp.array([n_edges]),
        nodes=None,
        edges={"k": jnp.ones(n_edges)},
        globals=None,
        senders=None,
        receivers=None,
    )


def test_padding_budget_covers_ddp_rank_window():
    """The padding budget for a train split must cover the rank-aligned
    window (a sliding window of ``batch_size``) that the DDP
    ``DistributedSampler`` hands to a rank -- not just the 0-aligned
    windows of the full split.

    With the 0-aligned-chunk implementation (the bug), the rank's first
    batch in this scenario exceeds the budget and ``jraph.pad_with_graphs``
    raises ``RuntimeError``.
    """
    # 10 distinct graphs; the tail (indices 5..9) is a cluster of larger
    # ones.  Under a 2-rank split with ``num_samples = ceil(10 / 2) = 5``,
    # rank 1 owns indices 5..9.  Its first batch (``batch_size = 4``) is
    # graphs 5..8.
    #
    # Edge sizes (chosen so that the rank window strictly exceeds the
    # 0-aligned window budget, mirroring the user-reported ``n_edge -2``):
    sizes = [1, 1, 1, 1, 1, 5, 5, 5, 3, 1]
    graphs = [_make_graph(s) for s in sizes]
    batch_size = 4

    padding = gcnn.data.GraphBatcher.calculate_padding(graphs, batch_size)

    # Rank 1's block and its first batch.
    num_replicas = 2
    num_samples = (len(graphs) + num_replicas - 1) // num_replicas  # ceil
    rank1_block = graphs[num_samples : num_samples * num_replicas]
    rank1_first_batch = rank1_block[:batch_size]

    # The first batch of rank 1 must fit inside the padding budget.
    actual_edges = sum(g.n_edge.item() for g in rank1_first_batch)
    assert actual_edges <= padding.n_edges, (
        f"Rank-1 first batch has {actual_edges} edges but the padding "
        f"budgets only {padding.n_edges}.  Under DDP this triggers "
        f"jraph.pad_with_graphs' RuntimeError("
        f"'Given graph is too large for the given padding').  "
        f"The padding must be the max over ALL windows of size "
        f"{batch_size}, not just the 0-aligned ones."
    )

    actual_nodes = sum(g.n_node.item() for g in rank1_first_batch)
    assert actual_nodes <= padding.n_nodes, (
        f"Rank-1 first batch has {actual_nodes} nodes but the padding "
        f"budgets only {padding.n_nodes}."
    )


def test_padding_budget_is_at_least_max_sliding_window():
    """``calculate_padding`` must return at least the max sum over every
    sliding window of ``batch_size`` in the dataset (for nodes and edges).
    This is the property the DDP rank-block partitioning depends on.
    """
    sizes = [2, 9, 1, 7, 4, 3, 5, 5, 5, 1]
    graphs = [_make_graph(s) for s in sizes]
    batch_size = 4

    pad = gcnn.data.GraphBatcher.calculate_padding(graphs, batch_size)

    max_nodes = 0
    max_edges = 0
    n = len(graphs)
    for start in range(max(1, n - batch_size + 1)):
        window = graphs[start : start + batch_size]
        max_nodes = max(max_nodes, sum(g.n_node.item() for g in window))
        max_edges = max(max_edges, sum(g.n_edge.item() for g in window))

    # jraph.pad_with_graphs requires at least ``sum(nodes) + 1`` nodes
    # (one dummy node for the padding graph), so the padding should be
    # >= max_nodes + 1 in a safe configuration. We test the lower bound
    # that is actually the source of the overflow: padding >= max_edges.
    assert pad.n_edges >= max_edges, (
        f"Padding.n_edges={pad.n_edges} < max sliding-window "
        f"edges={max_edges}.  A rank window with that edge sum would "
        f"overflow at pad time."
    )
    assert pad.n_nodes >= max_nodes, (
        f"Padding.n_nodes={pad.n_nodes} < max sliding-window " f"nodes={max_nodes}."
    )
