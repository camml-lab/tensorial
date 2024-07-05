import abc

import e3nn_jax as e3j
import jax.numpy as jnp

import tensorial

from . import radials


class SphericalBasis(tensorial.Attr):
    """A set of spherical harmonics basis functions"""

    def __init__(self, l_max: int, p_val=1, p_arg=-1):
        self._l_max = l_max
        self._p_val = p_val
        self._p_arg = p_arg
        super().__init__(e3j.s2_irreps(l_max, p_val, p_arg))

    @property
    def l_max(self) -> int:
        return self._l_max

    @property
    def p_val(self) -> int:
        return self._p_val

    @property
    def p_arg(self) -> int:
        return self._p_arg

    def evaluate(self, x) -> e3j.IrrepsArray:
        """Evaluate the spherical harmonics at the passed values.

        Warning: It is assumed that the values are located on the unit sphere (i.e. normalised
        vectors), no check is made to enforce this.
        """
        # * 2
        # * math.sqrt(math.pi)
        return e3j.spherical_harmonics(self.irreps, x, normalize=True, normalization="integral")

    def create_tensor(self, value) -> jnp.array:
        return self.evaluate(value)


class RadialSphericalBasis(tensorial.Attr):
    """A combined basis of a set of radial functions and spherical harmonics"""

    def create_tensor(self, value: jnp.array) -> jnp.array:
        """Create the signal that represents the expansion of the signal function in this basis"""
        return self.evaluate(value)

    @abc.abstractmethod
    def evaluate(self, value):
        """Evaluate the basis at the passed value"""


class SimpleRadialSphericalBasis(RadialSphericalBasis):

    def __init__(self, radial: radials.RadialBasis, spherical: SphericalBasis):
        self.radial = radial
        self.spherical = spherical
        num_radials = len(self.radial)
        super().__init__(spherical.irreps.repeat(num_radials))

    def evaluate(self, value):
        """Evaluate the basis functions at the given value"""
        angular = self.spherical.evaluate(value).array
        r = jnp.linalg.norm(value, axis=-1)
        radial = self.radial.evaluate(r)
        return jnp.einsum("...i,...j->...ij", radial, angular)

    def expand(self, x: jnp.array, coefficients: jnp.array):
        basis_values = self.evaluate(x)
        return jnp.einsum("ij,...ij->...", coefficients, basis_values)
