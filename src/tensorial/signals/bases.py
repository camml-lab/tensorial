"""Bases for spherical and radial-spherical expansions."""

import abc

import e3nn_jax as e3j
import jax
import jax.numpy as jnp
from typing_extensions import override

from . import radials
from .. import base


class SphericalBasis(base.Attr):
    """A set of spherical harmonics basis functions"""

    def __init__(self, l_max: int, p_val=1, p_arg=-1):
        """Initialise with the maximum degree and (optionally) parity constraints.

        Args:
            l_max: maximum degree of the harmonics.
            p_val: overall parity of the basis (1 for even, -1 for odd), or 0 for unconstrained.
            p_arg: parity of the individual irreps, or 0 for unconstrained.
        """
        self._l_max = l_max
        self._p_val = p_val
        self._p_arg = p_arg
        super().__init__(e3j.s2_irreps(l_max, p_val, p_arg))

    @property
    def l_max(self) -> int:
        """The maximum degree of the harmonics."""
        return self._l_max

    @property
    def p_val(self) -> int:
        """The overall parity value of the basis."""
        return self._p_val

    @property
    def p_arg(self) -> int:
        """The parity of the individual irreps in the basis."""
        return self._p_arg

    def evaluate(self, x) -> e3j.IrrepsArray:
        """Evaluate the spherical harmonics at the passed values.

        Warning: It is assumed that the values are located on the unit sphere (i.e. normalised
        vectors), no check is made to enforce this.
        """
        # * 2
        # * math.sqrt(math.pi)
        return e3j.spherical_harmonics(self.irreps, x, normalize=True, normalization="integral")

    @override
    def create_tensor(self, value) -> jax.Array:
        """Create the tensor (expansion coefficients) for ``value`` in this basis."""
        return self.evaluate(value)


class RadialSphericalBasis(base.Attr):
    """A combined basis of a set of radial functions and spherical harmonics."""

    @override
    def create_tensor(self, value: jax.Array) -> jax.Array:
        """Create the signal expansion coefficients for the value in this basis."""
        return self.evaluate(value)

    @abc.abstractmethod
    def evaluate(self, value):
        """Evaluate the basis at the passed value."""


class SimpleRadialSphericalBasis(RadialSphericalBasis):
    """A simple product basis: each radial function times each spherical harmonic."""

    def __init__(self, radial: radials.RadialBasis, spherical: SphericalBasis):
        """Combine a radial basis with a spherical basis.

        Args:
            radial: the radial basis to use.
            spherical: the spherical basis to use.
        """
        self.radial = radial
        self.spherical = spherical
        num_radials = len(self.radial)
        super().__init__(spherical.irreps.repeat(num_radials))

    @override
    def evaluate(self, value):
        """Evaluate the basis functions at the given value."""
        angular = self.spherical.evaluate(value).array
        r = jnp.linalg.norm(value, axis=-1)
        radial = self.radial.evaluate(r)
        return jnp.einsum("...i,...j->...ij", radial, angular)

    def expand(self, x: jax.Array, coefficients: jax.Array):
        """Reconstruct a scalar field from its basis coefficients at ``x``.

        Args:
            x: the point(s) at which to evaluate.
            coefficients: the expansion coefficients, shape ``(num_radials * l_dim, ...)``.

        Returns:
            the evaluated scalar field at each point.
        """
        basis_values = self.evaluate(x)
        return jnp.einsum("ij,...ij->...", coefficients, basis_values)
