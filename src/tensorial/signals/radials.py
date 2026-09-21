"""Radial basis functions (1D bases for the radial coordinate)."""

import abc
from collections.abc import Callable
import math

import e3nn_jax as e3j
import jax
import jax.numpy as jnp
from jaxtyping import Array
from typing_extensions import override

import tensorial


class RadialBasis(tensorial.Attr[Array]):
    """A set of radial basis functions.

    Attributes:
        number: the number of radial functions in the basis.
        domain: the ``(start, end)`` radial domain the basis spans.
    """

    _number: int
    _domain: tuple[float, float]

    def __init__(self, number: int, domain=(0.0, jnp.inf)):
        """Initialise with the number of radial functions and a radial domain.

        Args:
            number: the number of radial functions in the basis.
            domain: the ``(start, end)`` radial domain (defaults to ``[0, inf)``).
        """
        self._number = number
        self._domain = domain
        super().__init__(number * e3j.Irrep(0, p=1))

    @property
    def number(self) -> int:
        """Get the number of radial functions in the basis"""
        return self._number

    def __len__(self):
        """The total number of radial functions"""
        return self._number

    # @abc.abstractmethod
    # def __getitem__(self, n: int) -> Callable:
    #     """Get the radial function with index `n`"""

    @property
    def domain(self) -> tuple[float, float]:
        """The ``(start, end)`` radial domain the basis spans."""
        return self._domain

    @abc.abstractmethod
    def evaluate(self, radius):
        """Evaluate the radial basis at the given radius (or batch of radii)."""

    @override
    def create_tensor(self, value) -> e3j.IrrepsArray:
        """Wrap the evaluated coefficients into an :class:`e3nn_jax.IrrepsArray`."""
        return super().create_tensor(self.evaluate(value))  # pylint: disable=not-callable


class E3nnRadial(RadialBasis):
    """Select a radial function from the one-hot linspace built into e3nn-jax.

    see: https://e3nn-jax.readthedocs.io/en/latest/api/radial.html
    """

    _basis: Callable[[float], jnp.array]
    _cutoff: float

    def __init__(self, basis: str, max_radius: float, number: int, *, cutoff=None, min_radius=0.0):
        """Wrap one of e3nn-jax's soft-one-hot radial bases.

        Args:
            basis: e3nn-jax basis name (e.g. ``"gaussians"``, ``"bernstein"``, ...).
            max_radius: upper bound of the domain over which the basis is defined.
            number: number of radial functions.
            cutoff: optional cutoff flag/radius forwarded to e3nn-jax.
            min_radius: lower bound of the domain (default ``0.0``).
        """
        super().__init__(number, domain=(min_radius, max_radius))
        self._basis = basis
        self._cutoff = cutoff

    @property
    def basis(self) -> str:
        """The name of the e3nn-jax radial basis (e.g. `"gaussians"`, `"bernstein"`, ...)."""
        return self._basis

    @property
    def cutoff(self) -> bool | None:
        """Whether a cutoff should be applied, and if so its radius (or ``None``)."""
        return self._cutoff

    @override
    def evaluate(self, radius):
        """Evaluate the selected e3nn-jax soft-one-hot basis at the given radius (or radii)."""
        return e3j.soft_one_hot_linspace(
            radius,
            start=self.domain[0],
            end=self.domain[1],
            number=self.number,
            basis=self.basis,
            cutoff=self.cutoff,
        )


class E3nnPolyEnvelope(RadialBasis):
    """Polynomial envelope that can be used to make a radial basis smoothly approach zero at the
    cutoff.

    The envelope multiplies each radial sample by a smooth polynomial that is 1 below the
    smoothing start and decays to 0 by the end of the basis's domain, so the
    resulting radial functions taper to zero instead of being sharply cutoff.
    """

    _radials: RadialBasis
    _smoothing_start: float
    _smoothing_width: float
    _envelope: Callable[[float], jax.Array]

    def __init__(self, basis: RadialBasis, smoothing_start: float, n0: int, n1: int):
        """Wrap an existing :class:`RadialBasis` with a polynomial envelope.

        Args:
            basis: the underlying radial basis to wrap.
            smoothing_start: the radius at which the envelope begins to taper.
            n0: polynomial order of the envelope at its start (continuity condition).
            n1: polynomial order of the envelope at its end (continuity condition).

        Raises:
            ValueError: if ``smoothing_start`` is not inside the basis's domain.
        """
        super().__init__(basis.number, basis.domain)

        if smoothing_start < basis.domain[0] or smoothing_start > basis.domain[1]:
            raise ValueError(
                f"The start of the smoothing envelope ({smoothing_start}) must be within the "
                f"domain ({basis.domain}) of the radial basis"
            )

        self._radials = basis
        self._smoothing_start = smoothing_start
        self._smoothing_width = basis.domain[1] - smoothing_start
        self._envelope = e3j.poly_envelope(n0, n1, self._smoothing_width)

    def evaluate(self, radius):
        """Evaluate the underlying radial basis, then apply the polynomial envelope.

        All samples below ``smoothing_start`` are returned unchanged; all
        samples at or above it are scaled by the (per-sample) envelope value.
        """
        values = self._radials.evaluate(radius)
        mask = radius >= self._smoothing_start
        # Calculate the envelope for r values in the masked range
        envelope = self._envelope(radius[mask] - self._smoothing_start)
        # Multiply the values by the envelope, expanding the envelope to repeat by the number of
        # radials
        values = values.at[mask].set(
            values[mask, :] * envelope[:, jnp.newaxis].repeat(self.number, axis=1)
        )
        return values


class OrthoBasis(RadialBasis):
    """A radial basis orthonormalised on a set of discrete radial samples.

    The underlying (possibly non-orthogonal) radial basis is evaluated on a
    uniform grid; Gram-Schmidt orthonormalisation is then applied (using
    trapezoidal integration against the spherical surface measure) and the
    resulting orthonormal functions are linearly interpolated when evaluated
    at arbitrary radii.
    """

    radial_samples: jax.Array
    radial_step: jax.Array
    area_samples: jax.Array
    f_samples: jax.Array

    def __init__(self, radials: RadialBasis, n_samples: int):
        """Orthogonalise a radial basis on a uniform grid of ``n_samples`` points.

        Args:
            radials: the underlying radial basis to orthogonalise.
            n_samples: number of sample points to use for the Gram-Schmidt step.
        """
        super().__init__(radials.number, radials.domain)

        self.radial_samples = jnp.linspace(radials.domain[0], radials.domain[1], n_samples)
        self.radial_step = self.radial_samples[1] - self.radial_samples[0]

        non_orthogonal_samples = radials.evaluate(self.radial_samples)

        self.area_samples = 4 * math.pi * self.radial_samples * self.radial_samples
        self.f_samples = jnp.zeros_like(non_orthogonal_samples)

        u0 = non_orthogonal_samples[:, 0]
        self.f_samples = self.f_samples.at[:, 0].set(u0 / self.norm(u0))
        # self.f_samples[:, 0] = u0 / self.norm(u0)

        for i in range(1, self.number):
            ui = non_orthogonal_samples[:, i]
            for j in range(i):
                uj = self.f_samples[:, j]
                ui -= self.inner_product(uj, ui) / self.inner_product(uj, uj) * uj

            self.f_samples = self.f_samples.at[:, i].set(ui / self.norm(ui))

    @override
    def evaluate(self, radius):
        """Linearly interpolate the orthonormalised basis at the given radius (or radii)."""
        r_normalized = radius / self.radial_step
        r_normalized_floor_int = jnp.floor(r_normalized).astype(jnp.int64)
        # Get the indices of the samples just below the values of r
        indices_low = jnp.minimum(r_normalized_floor_int, jnp.array([len(self.radial_samples) - 2]))

        # Get what fraction through the samples we should be at
        r_remainder_normalized = r_normalized - indices_low
        r_remainder_normalized = r_remainder_normalized[:, jnp.newaxis].repeat(self.number, axis=1)

        low_samples = self.f_samples[indices_low, :]
        high_samples = self.f_samples[indices_low + 1, :]

        ret = low_samples * (1 - r_remainder_normalized) + high_samples * r_remainder_normalized

        return ret

    def inner_product(self, val_a, val_b):
        """Discrete inner product of two radial functions (trapezoidal rule).

        Args:
            val_a: First radial function, sampled at ``self.radial_samples``.
            val_b: Second radial function, sampled at the same points.

        Returns:
            The scalar inner product ``∫ f·g dr`` computed numerically.
        """
        return jnp.trapezoid(val_a * val_b * self.area_samples, self.radial_samples)

    def norm(self, val):
        """Discrete L2 norm of a radial function.

        Args:
            val: Radial function sampled at ``self.radial_samples``.

        Returns:
            The scalar ``sqrt(<val, val>)``.
        """
        return jnp.sqrt(self.inner_product(val, val))
