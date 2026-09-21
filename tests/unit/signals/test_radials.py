import jax.numpy as jnp
import numpy as np
import pytest

from tensorial import signals
from tensorial.signals import radials


class _PolynomialRadials(radials.RadialBasis):
    """A simple, deterministic radial basis: the first ``number`` powers of the radius."""

    def evaluate(self, radius):
        radius = jnp.atleast_1d(radius)
        return jnp.column_stack([radius**k for k in range(self.number)])


@pytest.fixture
def base_basis() -> radials.RadialBasis:
    return _PolynomialRadials(4, domain=(0.0, 10.0))


def test_radial_basis_properties(base_basis):
    assert base_basis.number == 4
    assert len(base_basis) == 4
    assert base_basis.domain == (0.0, 10.0)
    # A radial basis always carries scalar (0e) irreps, one per radial function.
    assert str(base_basis.irreps) == "4x0e"


def test_radial_basis_evaluate_shape(base_basis):
    radii = jnp.array([1.0, 2.0, 4.0])
    values = base_basis.evaluate(radii)
    assert values.shape == (3, 4)
    # column k is r**k
    np.testing.assert_allclose(values[2], [1.0, 4.0, 16.0, 64.0])


def test_e3nn_radial_properties():
    basis = radials.E3nnRadial("gaussians", max_radius=20.0, number=8, cutoff=19.0)
    assert basis.basis == "gaussians"
    assert basis.cutoff == 19.0
    assert basis.number == 8
    assert len(basis) == 8
    assert basis.domain == (0.0, 20.0)


@pytest.mark.parametrize("smoothing_start", [0.0, 5.0, 10.0])
def test_envelope_accepts_boundary_smoothing_starts(base_basis, smoothing_start):
    envelope = radials.E3nnPolyEnvelope(base_basis, smoothing_start, n0=3, n1=2)
    assert envelope.number == base_basis.number
    assert envelope.domain == base_basis.domain


@pytest.mark.parametrize("smoothing_start", [-1.0, 11.0])
def test_envelope_rejects_out_of_domain_smoothing_starts(base_basis, smoothing_start):
    with pytest.raises(ValueError):
        radials.E3nnPolyEnvelope(base_basis, smoothing_start, n0=3, n1=2)


def test_envelope_envelope_preserved_below_starting_point(base_basis):
    smoothing_start = 5.0
    envelope = radials.E3nnPolyEnvelope(base_basis, smoothing_start, n0=3, n1=2)

    below = jnp.array([0.0, 1.0, 2.5, 4.9])
    wrapped = envelope.evaluate(below)
    plain = base_basis.evaluate(below)
    # Samples strictly below the smoothing start are returned unchanged.
    np.testing.assert_allclose(wrapped, plain)


def test_envelope_unified_at_smoothing_start(base_basis):
    smoothing_start = 5.0
    envelope = radials.E3nnPolyEnvelope(base_basis, smoothing_start, n0=3, n1=2)

    # By construction the polynomial envelope evaluates to 1 at its start point,
    # so the wrapped basis should match the plain basis there.
    wrapped = envelope.evaluate(jnp.array([smoothing_start]))
    plain = base_basis.evaluate(jnp.array([smoothing_start]))
    np.testing.assert_allclose(wrapped, plain, rtol=1e-6, atol=1e-8)


def test_envelope_zero_at_domain_end(base_basis):
    smoothing_start = 5.0
    end = base_basis.domain[1]
    envelope = radials.E3nnPolyEnvelope(base_basis, smoothing_start, n0=3, n1=2)

    at_end = envelope.evaluate(jnp.array([end]))
    np.testing.assert_allclose(at_end, 0.0, atol=1e-6)


def test_orthobasis_orthonormal():
    num = 4
    ortho = radials.OrthoBasis(_PolynomialRadials(num, domain=(0.0, 10.0)), n_samples=400)

    samples = ortho.evaluate(ortho.radial_samples)
    gram = np.empty((num, num))
    for i in range(num):
        for j in range(num):
            gram[i, j] = float(ortho.inner_product(samples[:, i], samples[:, j]))

    # The orthonormalised basis is orthonormal under the basis's inner product.
    np.testing.assert_allclose(gram, np.eye(num), atol=1e-8)


def test_orthobasis_preserves_number_and_domain():
    base = _PolynomialRadials(5, domain=(0.0, 7.0))
    ortho = radials.OrthoBasis(base, n_samples=128)
    assert ortho.number == 5
    assert ortho.domain == (0.0, 7.0)
    assert len(ortho) == 5


def test_orthobasis_interpolates_stored_samples():
    ortho = radials.OrthoBasis(_PolynomialRadials(4, domain=(0.0, 10.0)), n_samples=200)

    # Between stored grid points the basis is the straight-line interpolation of the samples.
    interior_nodes = ortho.radial_samples[1:-1]
    values = ortho.evaluate(interior_nodes)
    np.testing.assert_allclose(values, ortho.f_samples[1:-1], atol=1e-10)


def test_orthobasis_interpolates_to_linear_between_nodes():
    ortho = radials.OrthoBasis(_PolynomialRadials(4, domain=(0.0, 10.0)), n_samples=200)

    step = ortho.radial_step
    midpoint = ortho.radial_samples[0] + step / 2
    value = ortho.evaluate(jnp.array([midpoint]))[0]
    expected = 0.5 * (ortho.f_samples[0] + ortho.f_samples[1])
    np.testing.assert_allclose(value, expected, atol=1e-10)


def test_signals_module_exports():
    assert signals.functions is not None
    assert signals.functions.DiracDelta is not None
