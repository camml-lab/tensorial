import functools

import e3nn_jax as e3j
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensorial.gcnn import _product_basis


def test_symmetric_contraction():
    num_types = 4
    x = e3j.normal("0e + 0o + 1o + 1e + 2e + 2o", jax.random.PRNGKey(0), (32, 128))
    types = jax.random.randint(jax.random.PRNGKey(1), (32,), minval=0, maxval=num_types)

    contraction = _product_basis.SymmetricContraction(3, ["0e", "1o", "2e"], num_types=num_types)
    params = contraction.init(jax.random.PRNGKey(0), x, types)

    e3j.utils.assert_equivariant(
        functools.partial(contraction.apply, params, input_type=types), jax.random.PRNGKey(3), x
    )


def test_contract_per_order_sum_equals_contract():
    num_types = 4
    x = e3j.normal("0e + 0o + 1o + 1e + 2e + 2o", jax.random.PRNGKey(0), (32, 128))
    types = jax.random.randint(jax.random.PRNGKey(1), (32,), minval=0, maxval=num_types)

    contraction = _product_basis.SymmetricContraction(3, ["0e", "1o", "2e"], num_types=num_types)
    params = contraction.init(jax.random.PRNGKey(0), x, types)

    out_sum = contraction.apply(params, x, types, per_order=False)
    out_dict = contraction.apply(params, x, types, per_order=True)

    assert isinstance(out_dict, dict)
    assert set(out_dict.keys()) == {1, 2, 3}

    total = sum(out_dict.values())

    assert total.irreps == out_sum.irreps
    np.testing.assert_allclose(total.array, out_sum.array, rtol=1e-5, atol=1e-5)


def test_contract_per_order_equivariance():
    num_types = 4
    x = e3j.normal("0e + 0o + 1o + 1e + 2e + 2o", jax.random.PRNGKey(0), (32, 128))
    types = jax.random.randint(jax.random.PRNGKey(1), (32,), minval=0, maxval=num_types)

    contraction = _product_basis.SymmetricContraction(3, ["0e", "1o", "2e"], num_types=num_types)
    params = contraction.init(jax.random.PRNGKey(0), x, types)

    for order in [1, 2, 3]:

        def apply_fn(x, input_type=types, o=order):
            return contraction.apply(params, x, input_type, per_order=True)[o]

        e3j.utils.assert_equivariant(apply_fn, jax.random.PRNGKey(3), x)


def test_contract_per_order_no_types():
    x = e3j.normal("0e + 0o + 1o + 1e + 2e + 2o", jax.random.PRNGKey(0), (32, 128))

    contraction = _product_basis.SymmetricContraction(3, ["0e", "1o", "2e"], num_types=1)
    params = contraction.init(jax.random.PRNGKey(0), x, None)

    out_sum = contraction.apply(params, x, None, per_order=False)
    out_dict = contraction.apply(params, x, None, per_order=True)

    total = sum(out_dict.values())
    np.testing.assert_allclose(total.array, out_sum.array, rtol=1e-5, atol=1e-5)


# ============================================================
# JointProductBasisBlock: shapes, irreps, equivariance
# ============================================================


def test_joint_product_basis_basic_shapes():
    num_types = 3
    n_nodes = 4

    x = e3j.normal("8x0e + 8x1o + 8x2e", jax.random.PRNGKey(0), (n_nodes,))
    g = e3j.normal("8x0e + 8x1o", jax.random.PRNGKey(1), (n_nodes,))
    types = jax.random.randint(jax.random.PRNGKey(2), (n_nodes,), 0, num_types)

    target = e3j.Irreps("0e + 1o")
    block = _product_basis.JointProductBasisBlock(
        irreps_out=target,
        node_correlation_order=2,
        global_correlation_order=2,
        num_types=num_types,
    )
    params = block.init(jax.random.PRNGKey(3), x, g, types)
    out = block.apply(params, x, g, types)

    assert out.shape == (n_nodes, target.dim)
    assert out.irreps == target


def test_joint_product_basis_no_types():
    n_nodes = 4
    x = e3j.normal("2x0e + 2x1o", jax.random.PRNGKey(0), (n_nodes,))
    g = e3j.normal("2x0e + 2x1o", jax.random.PRNGKey(1), (n_nodes,))

    block = _product_basis.JointProductBasisBlock(
        irreps_out="0e + 1o",
        node_correlation_order=2,
        global_correlation_order=2,
        num_types=1,
    )
    params = block.init(jax.random.PRNGKey(2), x, g, None)
    out = block.apply(params, x, g, None)

    assert out.shape[0] == n_nodes


def test_joint_product_basis_equivariance():
    """The joint block must be O(3)-equivariant in both x and g."""
    num_types = 2
    n_nodes = 4
    x = e3j.normal("4x0e + 4x1o", jax.random.PRNGKey(0), (n_nodes,))
    g = e3j.normal("4x0e + 4x1o", jax.random.PRNGKey(1), (n_nodes,))
    types = jnp.array([0, 1, 0, 1])

    block = _product_basis.JointProductBasisBlock(
        irreps_out="0e + 1o",
        node_correlation_order=2,
        global_correlation_order=2,
        num_types=num_types,
    )
    params = block.init(jax.random.PRNGKey(2), x, g, types)

    def apply_fn(x, g):
        return block.apply(params, x, g, types)

    e3j.utils.assert_equivariant(apply_fn, jax.random.PRNGKey(3), x, g)


# ============================================================
# Independence: parameter structure and per-(a,b) contribution
# ============================================================


def test_joint_product_basis_parameter_structure():
    """Every (a, b) pair must own a distinct parameter subtree."""
    num_types = 2
    n_nodes = 4
    x = e3j.normal("4x0e + 4x1o", jax.random.PRNGKey(0), (n_nodes,))
    g = e3j.normal("4x0e + 4x1o", jax.random.PRNGKey(1), (n_nodes,))
    types = jnp.array([0, 1, 0, 1])

    block = _product_basis.JointProductBasisBlock(
        irreps_out="0e + 1o",
        node_correlation_order=2,
        global_correlation_order=2,
        num_types=num_types,
    )
    params = block.init(jax.random.PRNGKey(2), x, g, types)

    paths: set = set()

    def collect(path, _):
        paths.add("/".join(str(k.key) for k in path))

    jax.tree_util.tree_map_with_path(collect, params)

    for a in range(3):
        for b in range(3):
            if a == 0 and b == 0:
                continue
            assert any(f"out_{a}_{b}/" in p for p in paths), f"missing out_{a}_{b}"
            if a > 0:
                assert any(f"x_{a}_{b}/" in p for p in paths), f"missing x_{a}_{b}"
            if b > 0:
                assert any(f"g_{a}_{b}/" in p for p in paths), f"missing g_{a}_{b}"


def test_joint_product_basis_isolated_atom():
    """Isolated nodes (x = 0) still receive a contribution from global-only terms."""
    num_types = 2
    n_nodes = 2

    # Node 0 is isolated: zero out its local features
    x = e3j.normal("4x0e + 4x1o", jax.random.PRNGKey(0), (n_nodes,))
    x = e3j.IrrepsArray(x.irreps, x.array.at[0].set(0.0))

    # Two different global fields, to check dependence
    g1 = e3j.normal("4x0e + 4x1o", jax.random.PRNGKey(1), (n_nodes,))
    g2 = e3j.normal("4x0e + 4x1o", jax.random.PRNGKey(2), (n_nodes,))

    types = jnp.array([0, 1])
    block = _product_basis.JointProductBasisBlock(
        irreps_out="0e",
        node_correlation_order=2,
        global_correlation_order=2,
        num_types=num_types,
    )
    params = block.init(jax.random.PRNGKey(3), x, g1, types)

    out1 = block.apply(params, x, g1, types)
    out2 = block.apply(params, x, g2, types)

    # 1. Isolated node's output must be finite and non-zero (driven by g)
    assert jnp.isfinite(out1.array[0]).all()
    assert not jnp.allclose(out1.array[0], 0.0)

    # 2. It must depend on the global field
    assert not jnp.allclose(out1.array[0], out2.array[0])

    # 3. It must differ from the normal node's output
    assert not jnp.allclose(out1.array[0], out1.array[1])


# ============================================================
# Sanity: joint block degenerates correctly
# ============================================================


def test_joint_product_basis_requires_nonzero_orders():
    """(0, 0) alone is invalid — there is nothing to compute."""
    with pytest.raises(ValueError):
        block = _product_basis.JointProductBasisBlock(
            irreps_out="0e",
            node_correlation_order=0,
            global_correlation_order=0,
            num_types=1,
        )
        x = e3j.normal("2x0e", jax.random.PRNGKey(0), (2,))
        block.init(jax.random.PRNGKey(1), x, x, None)


def _path_str(path):
    return "/".join(str(k.key) for k in path)


def _collect_paths(params) -> set[str]:
    paths = set()
    jax.tree_util.tree_map_with_path(lambda p, _: paths.add(_path_str(p)), params)
    return paths


def _zero_subtree(params, marker: str):
    """Zero every leaf whose path contains `marker` (e.g., '/out_0_1/')."""

    def f(path, leaf):
        if marker in _path_str(path):
            return jnp.zeros_like(leaf)
        return leaf

    return jax.tree_util.tree_map_with_path(f, params)


def test_joint_product_basis_all_terms_active():
    """Zeroing any (a, b)'s parameter subtree changes the output.

    Skips pairs whose output Linear is genuinely parameter-free (matching input/output
    irreps with multiplicity 1); those cannot be tested by zeroing.
    """
    num_types = 2
    n_nodes = 4
    x = e3j.normal("4x0e + 4x1o", jax.random.PRNGKey(0), (n_nodes,))
    g = e3j.normal("4x0e + 4x1o", jax.random.PRNGKey(1), (n_nodes,))
    types = jnp.array([0, 1, 0, 1])

    block = _product_basis.JointProductBasisBlock(
        irreps_out="0e + 1o",
        node_correlation_order=2,
        global_correlation_order=2,
        num_types=num_types,
    )
    params = block.init(jax.random.PRNGKey(2), x, g, types)
    baseline = block.apply(params, x, g, types)

    all_paths = _collect_paths(params)

    checked_any = False
    for a in range(3):
        for b in range(3):
            if a == 0 and b == 0:
                continue

            marker = f"/out_{a}_{b}/"
            if not any(marker in p for p in all_paths):
                # No learnable parameters for this pair's output linear —
                # nothing to zero, so we can't test structural activity this way.
                continue

            checked_any = True
            params_a = _zero_subtree(params, marker)
            out = block.apply(params_a, x, g, types)
            assert not np.allclose(baseline.array, out.array), f"zeroing {marker} had no effect"

    assert checked_any, "no (a, b) pair had parameters to test"


def test_joint_product_basis_output_linear_has_learnable_params():
    """Every (a, b) output Linear must contribute learnable parameters."""
    num_types = 2
    n_nodes = 4
    x = e3j.normal("4x0e + 4x1o", jax.random.PRNGKey(0), (n_nodes,))
    g = e3j.normal("4x0e + 4x1o", jax.random.PRNGKey(1), (n_nodes,))
    types = jnp.array([0, 1, 0, 1])

    block = _product_basis.JointProductBasisBlock(
        irreps_out="0e + 1o",
        node_correlation_order=2,
        global_correlation_order=2,
        num_types=num_types,
    )
    params = block.init(jax.random.PRNGKey(2), x, g, types)

    # Build a summary per pair: number of leaves, total parameter count.
    summary: dict[tuple[int, int], dict] = {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(params):
        s = "/".join(str(k.key) for k in path)
        for a in range(3):
            for b in range(3):
                if a == 0 and b == 0:
                    continue
                marker = f"/out_{a}_{b}/"
                if marker in s:
                    info = summary.setdefault((a, b), {"n_leaves": 0, "n_params": 0, "paths": []})
                    info["n_leaves"] += 1
                    info["n_params"] += int(jnp.size(leaf))
                    info["paths"].append(s)

    for (a, b), info in sorted(summary.items()):
        print(f"({a}, {b}): {info['n_leaves']} leaves, " f"{info['n_params']} params")
        for p in info["paths"]:
            print(f"    {p}")

    # Assertion: each pair must have at least one learnable parameter.
    for a in range(3):
        for b in range(3):
            if a == 0 and b == 0:
                continue
            assert (a, b) in summary, f"({a}, {b}) has no parameters at all"
            assert summary[(a, b)]["n_params"] > 0, f"({a}, {b}) has no learnable parameters"
