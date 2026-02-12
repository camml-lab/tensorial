import jax
import jax.numpy as jnp
import pytest

from tensorial.gcnn import graph_ops


def test_segment_mean_basic():
    # 2 segments, sizes 3 and 2
    segment_sizes = jnp.array([3, 2])
    # Data: [0, 1, 2,  3, 4]
    data = jnp.arange(5, dtype=jnp.float32)

    # Expected:
    # Seg 0: (0+1+2)/3 = 1.0
    # Seg 1: (3+4)/2 = 3.5
    expected = jnp.array([1.0, 3.5])

    res = graph_ops.segment_mean(data, segment_sizes)
    assert jnp.allclose(res, expected)


def test_segment_mean_masked_data():
    # 1 segment, size 4
    segment_sizes = jnp.array([4])
    # Data: [10, 20, 30, 40]
    data = jnp.array([10.0, 20.0, 30.0, 40.0])
    # Mask: [T, T, F, F] -> Keep 10, 20
    mask = jnp.array([True, True, False, False])

    # Expected: (10+20)/2 = 15.0

    res = graph_ops.segment_mean(data, segment_sizes, mask=mask)
    assert jnp.allclose(res, 15.0)


def test_segment_mean_empty_segment():
    # Segments: [2, 0, 2]
    segment_sizes = jnp.array([2, 0, 2])
    data = jnp.array([1.0, 2.0, 3.0, 4.0])
    # Seg 0: 1, 2 -> 1.5
    # Seg 1: empty -> 0.0 (safe_counts > 0 check)
    # Seg 2: 3, 4 -> 3.5

    res = graph_ops.segment_mean(data, segment_sizes)
    expected = jnp.array([1.5, 0.0, 3.5])
    assert jnp.allclose(res, expected)


def test_segment_mean_all_masked():
    # 1 segment, size 3
    segment_sizes = jnp.array([3])
    data = jnp.array([1.0, 2.0, 3.0])
    mask = jnp.array([False, False, False])

    # Validation: count is 0.
    # safe_counts > 0 check -> returns 0.
    res = graph_ops.segment_mean(data, segment_sizes, mask=mask)
    assert jnp.allclose(res, 0.0)


def test_segment_sum_with_mask():
    segment_sizes = jnp.array([2, 2])
    data = jnp.array([1.0, 2.0, 3.0, 4.0])
    # Mask for segments: 1st valid, 2nd invalid
    segment_mask = jnp.array([True, False])

    res = graph_ops.segment_sum(data, segment_sizes, segment_mask=segment_mask)
    # Seg 1: 1+2 = 3
    # Seg 2: invalid -> 0
    expected = jnp.array([3.0, 0.0])
    assert jnp.allclose(res, expected)


def test_segment_min_with_mask():
    segment_sizes = jnp.array([2, 2])
    data = jnp.array([1.0, 2.0, 3.0, 4.0])
    # Mask for segments: 1st valid, 2nd invalid
    segment_mask = jnp.array([True, False])

    res = graph_ops.segment_min(data, segment_sizes, segment_mask=segment_mask)
    # Seg 1: min(1, 2) = 1.0
    # Seg 2: invalid -> +inf
    expected = jnp.array([1.0, jnp.inf])
    assert jnp.allclose(res, expected)


def test_segment_max_with_mask():
    segment_sizes = jnp.array([2, 2])
    data = jnp.array([1.0, 2.0, 3.0, 4.0])
    # Mask for segments: 1st valid, 2nd invalid
    segment_mask = jnp.array([True, False])

    res = graph_ops.segment_max(data, segment_sizes, segment_mask=segment_mask)
    # Seg 1: max(1, 2) = 2.0
    # Seg 2: invalid -> -inf
    expected = jnp.array([2.0, -jnp.inf])
    assert jnp.allclose(res, expected)


def test_segment_reduce_interface():
    segment_sizes = jnp.array([2])
    data = jnp.array([1.0, 3.0])
    segment_mask = jnp.array([True])

    # Check all reductions pass through
    assert jnp.allclose(
        graph_ops.segment_reduce(data, segment_sizes, "mean", segment_mask=segment_mask), 2.0
    )
    assert jnp.allclose(
        graph_ops.segment_reduce(data, segment_sizes, "sum", segment_mask=segment_mask), 4.0
    )
    assert jnp.allclose(
        graph_ops.segment_reduce(data, segment_sizes, "min", segment_mask=segment_mask), 1.0
    )
    assert jnp.allclose(
        graph_ops.segment_reduce(data, segment_sizes, "max", segment_mask=segment_mask), 3.0
    )


def test_segment_reduce_broadcasting():
    # Test broadcasting of segment_mask (B,) against data result (B, D)
    segment_sizes = jnp.array([2, 2])
    # data has feature dim
    data = jnp.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])  # (4, 2)
    segment_mask = jnp.array([True, False])  # (2,)

    res = graph_ops.segment_mean(data, segment_sizes, segment_mask=segment_mask)
    # Seg 1: mean([1,10], [2,20]) = [1.5, 15.0]
    # Seg 2: invalid -> [0, 0]

    expected = jnp.array([[1.5, 15.0], [0.0, 0.0]])
    assert jnp.allclose(res, expected)

    # Test min broadcasting
    res_min = graph_ops.segment_min(data, segment_sizes, segment_mask=segment_mask)
    expected_min = jnp.array([[1.0, 10.0], [jnp.inf, jnp.inf]])
    assert jnp.allclose(res_min, expected_min)


def test_segment_mean_integration(cube_graph):
    # Integration test using the fixture
    nodes = jnp.arange(8, dtype=jnp.float32)
    n_node = cube_graph.n_node  # [8]

    res = graph_ops.segment_mean(nodes, n_node)
    assert jnp.allclose(res, 3.5)


def test_jit_compilation():
    # Verify that segment_mean (and implicitly others) can be jitted
    segment_sizes = jnp.array([3, 2])
    data = jnp.arange(5, dtype=jnp.float32)
    segment_mask = jnp.array([True, False])

    @jax.jit
    def jitted_mean(d, s, m):
        return graph_ops.segment_mean(d, s, segment_mask=m)

    res = jitted_mean(data, segment_sizes, segment_mask)
    # Seg 0: 1.0
    # Seg 1: invalid -> 0.0
    expected = jnp.array([1.0, 0.0])
    assert jnp.allclose(res, expected)
