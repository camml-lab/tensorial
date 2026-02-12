import jax
import jax.numpy as jnp
import pytest

from tensorial.gcnn import graph_ops


@pytest.mark.parametrize("jit", [False, True])
def test_segment_mean_basic(jit):
    op = jax.jit(graph_ops.segment_mean) if jit else graph_ops.segment_mean

    # 2 segments, sizes 3 and 2
    segment_sizes = jnp.array([3, 2])
    # Data: [0, 1, 2,  3, 4]
    data = jnp.arange(5, dtype=jnp.float32)

    # Expected:
    # Seg 0: (0+1+2)/3 = 1.0
    # Seg 1: (3+4)/2 = 3.5
    expected = jnp.array([1.0, 3.5])

    res = op(data, segment_sizes)
    assert jnp.allclose(res, expected)


@pytest.mark.parametrize("jit", [False, True])
def test_segment_mean_masked_data(jit):
    op = jax.jit(graph_ops.segment_mean) if jit else graph_ops.segment_mean

    # 1 segment, size 4
    segment_sizes = jnp.array([4])
    # Data: [10, 20, 30, 40]
    data = jnp.array([10.0, 20.0, 30.0, 40.0])
    # Mask: [T, T, F, F] -> Keep 10, 20
    mask = jnp.array([True, True, False, False])

    # Expected: (10+20)/2 = 15.0

    res = op(data, segment_sizes, mask=mask)
    assert jnp.allclose(res, 15.0)


@pytest.mark.parametrize("jit", [False, True])
def test_segment_mean_empty_segment(jit):
    op = jax.jit(graph_ops.segment_mean) if jit else graph_ops.segment_mean

    # Segments: [2, 0, 2]
    segment_sizes = jnp.array([2, 0, 2])
    data = jnp.array([1.0, 2.0, 3.0, 4.0])
    # Seg 0: 1, 2 -> 1.5
    # Seg 1: empty -> 0.0 (safe_counts > 0 check)
    # Seg 2: 3, 4 -> 3.5

    res = op(data, segment_sizes)
    expected = jnp.array([1.5, 0.0, 3.5])
    assert jnp.allclose(res, expected)


@pytest.mark.parametrize("jit", [False, True])
def test_segment_mean_all_masked(jit):
    op = jax.jit(graph_ops.segment_mean) if jit else graph_ops.segment_mean

    # 1 segment, size 3
    segment_sizes = jnp.array([3])
    data = jnp.array([1.0, 2.0, 3.0])
    mask = jnp.array([False, False, False])

    # Validation: count is 0.
    # safe_counts > 0 check -> returns 0.
    res = op(data, segment_sizes, mask=mask)
    assert jnp.allclose(res, 0.0)


@pytest.mark.parametrize("jit", [False, True])
def test_segment_sum_with_mask(jit):
    op = jax.jit(graph_ops.segment_sum) if jit else graph_ops.segment_sum

    segment_sizes = jnp.array([2, 2])
    data = jnp.array([1.0, 2.0, 3.0, 4.0])
    # Mask for segments: 1st valid, 2nd invalid
    segment_mask = jnp.array([True, False])

    res = op(data, segment_sizes, segment_mask=segment_mask)
    # Seg 1: 1+2 = 3
    # Seg 2: invalid -> 0
    expected = jnp.array([3.0, 0.0])
    assert jnp.allclose(res, expected)


@pytest.mark.parametrize("jit", [False, True])
def test_segment_min_with_mask(jit):
    op = jax.jit(graph_ops.segment_min) if jit else graph_ops.segment_min

    segment_sizes = jnp.array([2, 2])
    data = jnp.array([1.0, 2.0, 3.0, 4.0])
    # Mask for segments: 1st valid, 2nd invalid
    segment_mask = jnp.array([True, False])

    res = op(data, segment_sizes, segment_mask=segment_mask)
    # Seg 1: min(1, 2) = 1.0
    # Seg 2: invalid -> +inf
    expected = jnp.array([1.0, jnp.inf])
    assert jnp.allclose(res, expected)


@pytest.mark.parametrize("jit", [False, True])
def test_segment_max_with_mask(jit):
    op = jax.jit(graph_ops.segment_max) if jit else graph_ops.segment_max

    segment_sizes = jnp.array([2, 2])
    data = jnp.array([1.0, 2.0, 3.0, 4.0])
    # Mask for segments: 1st valid, 2nd invalid
    segment_mask = jnp.array([True, False])

    res = op(data, segment_sizes, segment_mask=segment_mask)
    # Seg 1: max(1, 2) = 2.0
    # Seg 2: invalid -> -inf
    expected = jnp.array([2.0, -jnp.inf])
    assert jnp.allclose(res, expected)


@pytest.mark.parametrize("jit", [False, True])
def test_segment_reduce_interface(jit):
    # For segment_reduce, we need to partial the static args
    base_op = graph_ops.segment_reduce
    if jit:
        op = jax.jit(base_op, static_argnames=("reduction",))
    else:
        op = base_op

    segment_sizes = jnp.array([2])
    data = jnp.array([1.0, 3.0])
    segment_mask = jnp.array([True])

    # Check all reductions pass through
    assert jnp.allclose(op(data, segment_sizes, "mean", segment_mask=segment_mask), 2.0)
    assert jnp.allclose(op(data, segment_sizes, "sum", segment_mask=segment_mask), 4.0)
    assert jnp.allclose(op(data, segment_sizes, "min", segment_mask=segment_mask), 1.0)
    assert jnp.allclose(op(data, segment_sizes, "max", segment_mask=segment_mask), 3.0)


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize(
    "reduction, expected",
    [
        ("mean", jnp.array([[1.5, 15.0], [0.0, 0.0]])),
        ("sum", jnp.array([[3.0, 30.0], [0.0, 0.0]])),
        ("min", jnp.array([[1.0, 10.0], [jnp.inf, jnp.inf]])),
        ("max", jnp.array([[2.0, 20.0], [-jnp.inf, -jnp.inf]])),
    ],
)
def test_segment_reduce_broadcasting(jit, reduction, expected):
    op = getattr(graph_ops, f"segment_{reduction}")
    if jit:
        op = jax.jit(op)

    # Test broadcasting of segment_mask (B,) against data result (B, D)
    segment_sizes = jnp.array([2, 2])
    # data has feature dim
    data = jnp.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])  # (4, 2)
    segment_mask = jnp.array([True, False])  # (2,)

    res = op(data, segment_sizes, segment_mask=segment_mask)
    assert jnp.allclose(res, expected)


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize(
    "reduction, expected",
    [
        ("mean", 3.5),
        ("sum", 28.0),
        ("min", 0.0),
        ("max", 7.0),
    ],
)
def test_segment_reduce_integration(cube_graph, jit, reduction, expected):
    op = getattr(graph_ops, f"segment_{reduction}")
    if jit:
        op = jax.jit(op)

    # Integration test using the fixture
    nodes = jnp.arange(8, dtype=jnp.float32)
    n_node = cube_graph.n_node  # [8]

    res = op(nodes, n_node)
    assert jnp.allclose(res, expected)


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize(
    "reduction, expected_inf",
    [
        ("min", jnp.inf),
        ("max", -jnp.inf),
    ],
)
def test_segment_reduce_with_explicit_inf(jit, reduction, expected_inf):
    op = getattr(graph_ops, f"segment_{reduction}")
    if jit:
        op = jax.jit(op)

    # 2 Segments, sizes [2, 2]
    segment_sizes = jnp.array([2, 2])
    data = jnp.array([1.0, 2.0, 3.0, 4.0])

    # Case 1: Masked at data level (resulting in all inf for a segment)
    mask = jnp.array([True, True, False, False])
    # Expected:
    # Seg 0: reduce(1, 2) = 1.0 (min) or 2.0 (max)
    # Seg 1: reduce(inf, inf) = expected_inf
    res = op(data, segment_sizes, mask=mask, segment_mask=None)
    expected_val = 1.0 if reduction == "min" else 2.0
    expected = jnp.array([expected_val, expected_inf])
    assert jnp.allclose(res, expected)

    # Case 2: Masked at segment level
    mask = jnp.array([True, True, True, True])
    segment_mask = jnp.array([True, False])

    res = op(data, segment_sizes, mask=mask, segment_mask=segment_mask)
    assert jnp.allclose(res, expected)
