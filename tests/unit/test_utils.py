import e3nn_jax as e3j
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensorial import utils


def test_infer_backend_numpy():
    assert utils.infer_backend(np.array([1.0, 2.0])) is np


def test_infer_backend_jax():
    assert utils.infer_backend(jnp.array([1.0, 2.0])) is jnp


def test_infer_backend_empty_pytree():
    assert utils.infer_backend(jax.tree_util.tree_map(lambda x: x, {})) is jnp


def test_infer_backend_mixed_raises():
    with pytest.raises(ValueError, match="Cannot mix numpy and jax arrays"):
        utils.infer_backend([np.array([1.0]), jnp.array([1.0])])


def test_zeros():
    irreps = e3j.Irreps("2o + 1e")
    result = utils.zeros(irreps)
    assert isinstance(result, e3j.IrrepsArray)
    assert result.irreps == irreps
    assert result.shape == (irreps.dim,)
    assert jnp.all(result.array == 0)
    assert result.zero_flags == (True,) * len(irreps)


def test_zeros_with_leading_shape_and_dtype():
    irreps = e3j.Irreps("1o")
    result = utils.zeros(irreps, leading_shape=(3, 4), dtype=jnp.float32)
    assert result.shape == (3, 4, irreps.dim)
    assert result.array.dtype == jnp.float32
    assert jnp.all(result.array == 0)


def test_zeros_numpy_backend():
    irreps = e3j.Irreps("1o")
    result = utils.zeros(irreps, np_=np)
    assert isinstance(result.array, np.ndarray)
    assert jnp.all(result.array == 0)


def test_zeros_like_jax():
    original = utils.ones(e3j.Irreps("2e"), leading_shape=(2,))
    result = utils.zeros_like(original)
    assert result.irreps == original.irreps
    assert result.shape == original.shape
    assert result.dtype == original.dtype
    assert jnp.all(result.array == 0)


def test_zeros_like_numpy():
    original = utils.ones(e3j.Irreps("1o"), np_=np)
    result = utils.zeros_like(original)
    assert isinstance(result.array, np.ndarray)
    assert jnp.all(result.array == 0)


def test_ones():
    irreps = e3j.Irreps("1o + 2e")
    result = utils.ones(irreps)
    assert isinstance(result, e3j.IrrepsArray)
    assert result.irreps == irreps
    assert result.shape == (irreps.dim,)
    assert jnp.all(result.array == 1)
    assert result.zero_flags == (False,) * len(irreps)


def test_ones_with_leading_shape_and_dtype():
    irreps = e3j.Irreps("1e")
    result = utils.ones(irreps, leading_shape=(5,), dtype=jnp.float64)
    assert result.shape == (5, irreps.dim)
    assert jnp.all(result.array == 1)


def test_ones_numpy_backend():
    irreps = e3j.Irreps("1o")
    result = utils.ones(irreps, np_=np)
    assert isinstance(result.array, np.ndarray)
    assert jnp.all(result.array == 1)


def test_ones_like_jax():
    original = utils.zeros(e3j.Irreps("3o"))
    result = utils.ones_like(original)
    assert result.irreps == original.irreps
    assert result.shape == original.shape
    assert result.dtype == original.dtype
    assert jnp.all(result.array == 1)


def test_ones_like_numpy():
    original = utils.zeros(e3j.Irreps("2e"), np_=np)
    result = utils.ones_like(original)
    assert isinstance(result.array, np.ndarray)
    assert jnp.all(result.array == 1)


def test_optional_import_success():
    mod = utils.optional_import("numpy")
    assert mod is np


def test_optional_import_missing_with_extra():
    with pytest.raises(ImportError, match="required for this feature"):
        utils.optional_import("nonexistent_module_xyz", extra="extra")


def test_optional_import_missing_without_extra():
    with pytest.raises(ImportError, match="required for this feature"):
        utils.optional_import("nonexistent_module_xyz")


@pytest.mark.parametrize(
    "parity, expected",
    [
        ("o + e", "8x0o + 8x0e + 8x1o + 8x1e + 8x2o + 8x2e"),
        ("e + o", "8x0e + 8x0o + 8x1e + 8x1o + 8x2e + 8x2o"),
        ("e", "8x0e + 8x1e + 8x2e"),
        ("o", "8x0o + 8x1o + 8x2o"),
        ("sh", "8x0e + 8x1o + 8x2e"),
    ],
)
def test_make_irreps(parity, expected):
    assert utils.make_irreps(8, 2, parity=parity) == e3j.Irreps(expected)


def test_make_irreps_defaults_to_both_parities_odd_first():
    assert utils.make_irreps(4, 1) == e3j.Irreps("4x0o + 4x0e + 4x1o + 4x1e")


def test_make_irreps_parity_ignores_spaces():
    assert utils.make_irreps(2, 1, parity="e+o") == utils.make_irreps(2, 1, parity="e + o")


def test_make_irreps_ell_max_zero():
    assert utils.make_irreps(3, 0) == e3j.Irreps("3x0o + 3x0e")
    assert utils.make_irreps(3, 0, parity="sh") == e3j.Irreps("3x0e")


def test_make_irreps_sh_matches_spherical_harmonics():
    assert utils.make_irreps(1, 4, parity="sh") == e3j.Irreps.spherical_harmonics(4)


def test_make_irreps_is_exported_at_top_level():
    import tensorial  # pylint: disable=import-outside-toplevel

    assert tensorial.make_irreps is utils.make_irreps


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(mul=0, ell_max=2), "'mul'"),
        (dict(mul=8, ell_max=-1), "'ell_max'"),
        (dict(mul=8, ell_max=2, parity="x"), "Unknown parity"),
    ],
)
def test_make_irreps_invalid(kwargs, match):
    with pytest.raises(ValueError, match=match):
        utils.make_irreps(**kwargs)
