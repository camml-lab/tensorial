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
