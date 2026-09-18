import functools
from typing import Final

import e3nn_jax as e3j
import jax
from jax import ops
import jax.numpy as jnp
import jraph
import numpy as np
import pytest

from tensorial import gcnn
from tensorial.gcnn import mace

from .. import utils


def get_dummy_graph(n_nodes=3, n_edges=4):
    return jraph.GraphsTuple(
        n_node=np.array([n_nodes]),
        n_edge=np.array([n_edges]),
        nodes={"features": np.ones((n_nodes, 2))},
        edges={"features": np.ones((n_edges, 2))},
        globals={"features": np.zeros((1, 2))},
        senders=np.arange(n_edges) % n_nodes,
        receivers=(np.arange(n_edges) + 1) % n_nodes,
    )


def test_graph_batcher_padding_to_multiple():
    graphs = [get_dummy_graph() for _ in range(5)]

    batcher_32 = gcnn.data.GraphBatcher(graphs, batch_size=5, pad=True, pad_to_multiple=32)
    batch_32 = next(iter(batcher_32))
    assert batch_32.nodes["features"].shape[0] == 32
    assert batch_32.edges["features"].shape[0] == 32

    batcher_gpu = gcnn.data.GraphBatcher(graphs, batch_size=5, pad=True, pad_to_multiple="gpu")
    batch_gpu = next(iter(batcher_gpu))
    assert batch_gpu.nodes["features"].shape[0] == 64
    assert batch_gpu.edges["features"].shape[0] == 64


@pytest.mark.benchmark
def test_graph_batcher_padding_simple(benchmark):
    # 1. SETUP: High-node count to make the benchmark meaningful
    n_graphs: Final[int] = 1277  # prime number
    graphs = [get_dummy_graph(n_nodes=50, n_edges=100) for _ in range(n_graphs)]

    # Define a JIT-compiled GNN-like function to measure real hardware execution
    @jax.jit
    def mock_message_passing(nodes, senders, receivers):
        # A simple linear message passing: sum of sender nodes
        messages = nodes[senders]
        return ops.segment_sum(messages, receivers, num_segments=nodes.shape[0])

    # 2. PREPARE DATA: Create the padded batch once outside the loop
    # We pad to 1024 to ensure we hit a power-of-two/multiple-of-128 boundary
    batcher_padded = gcnn.data.GraphBatcher(
        graphs, batch_size=n_graphs, pad=True, pad_to_multiple=1024
    )
    batch = next(iter(batcher_padded))

    # Convert dictionary features to JAX arrays and move to device
    nodes = jax.device_put(jnp.array(batch.nodes["features"]))
    senders = jax.device_put(jnp.array(batch.senders))
    receivers = jax.device_put(jnp.array(batch.receivers))

    # 3. WARMUP: Eliminate the "System Call" spike from JIT compilation
    _ = mock_message_passing(nodes, senders, receivers).block_until_ready()

    # 4. BENCHMARK: Measure the execution of the equivariant operation
    # Using block_until_ready() ensures we measure the actual GPU/TPU time
    def run_op():
        out = mock_message_passing(nodes, senders, receivers)
        return out.block_until_ready()

    result_padded = benchmark(run_op)

    # 5. FINAL VERIFICATION: Ensure padding didn't break the math
    # (Optional: only runs once after the benchmark iterations)
    num_real_nodes = sum(g.n_node[0] for g in graphs)
    # Validate against a non-padded baseline
    batcher_unpadded = gcnn.data.GraphBatcher(graphs, batch_size=n_graphs, pad=False)
    batch_raw = next(iter(batcher_unpadded))
    expected = mock_message_passing(
        jnp.array(batch_raw.nodes["features"]),
        jnp.array(batch_raw.senders),
        jnp.array(batch_raw.receivers),
    )

    np.testing.assert_allclose(expected, result_padded[:num_real_nodes], atol=1e-5)


@pytest.mark.benchmark
def test_graph_batcher_padding_benchmark(benchmark):
    graphs = [get_dummy_graph(n_nodes=10, n_edges=20) for _ in range(32)]

    @benchmark
    def batch_with_padding():
        batcher = gcnn.data.GraphBatcher(graphs, batch_size=32, pad=True, pad_to_multiple=64)
        list(batcher)


@pytest.mark.parametrize("pad_multiple", [None, 32, 64, 128, 256, 512])
@pytest.mark.parametrize("bitwidth", [16, 32, 64])
@pytest.mark.benchmark
def test_benchmark_message_passing(cube_graph, pad_multiple, bitwidth, benchmark, record_property):
    n_graphs: Final[int] = 12  # 1277  # prime number
    device = jax.devices()[0]

    int_dtype = getattr(jnp, f"int{bitwidth}")
    float_dtype = getattr(jnp, f"float{bitwidth}", None)
    if float_dtype is None and bitwidth == 8:
        # Fallback for 8-bit float if available
        float_dtype = getattr(jnp, "float8_e4m3fn", None)

    def _cast(x):
        if hasattr(x, "dtype"):
            if jnp.issubdtype(x.dtype, jnp.integer):
                # Only cast integer if it's not n_node or n_edge, to avoid early overflow before batching?
                # Actually, jraph might promote during padding/batching. Let's cast all.
                return x.astype(int_dtype)
            if jnp.issubdtype(x.dtype, jnp.floating) and float_dtype is not None:
                return x.astype(float_dtype)
        return x

    cube_graph = jax.tree.map(_cast, cube_graph)

    dataset = [cube_graph] * n_graphs
    batcher = gcnn.data.GraphBatcher(dataset, batch_size=27, pad=True, pad_to_multiple=pad_multiple)

    r_max = 5.0
    num_types = 3

    model = utils.graph_model(
        r_max,
        e3j.Irreps("0e + 1o + 2e"),
        mace.Mace(
            irreps_out=e3j.Irreps("0e"),
            out_field=gcnn.atomic.ENERGY_PER_ATOM,
            hidden_irreps="2x0e + 2x1o",
            num_types=num_types,
            y0_values=np.random.rand(num_types).tolist(),
        ),
        type_numbers=[0],
    )

    # Inject into the log
    record_property("n_nodes", batcher.padding.n_nodes)
    record_property("n_edges", batcher.padding.n_edges)
    record_property("n_graphs", batcher.padding.n_graphs)

    # Pre-materialize the batches so we don't measure the cost of the batching itself
    batches = list(batcher)
    batches = jax.device_put(batches, device)

    example_patch = batches[0]
    params = model.init(jax.random.PRNGKey(0), example_patch)
    params = jax.tree.map(_cast, params)
    apply = functools.partial(jax.jit(model.apply), params)

    # Do a JIT warmup
    jax.block_until_ready(apply(example_patch))

    @benchmark
    def do_bench():
        out = None
        for batch in batches:
            out = apply(batch)

        if out is not None:
            # Make sure we force JAX to finish and flush the result
            jax.block_until_ready(out)
