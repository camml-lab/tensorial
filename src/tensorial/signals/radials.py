import abc
from collections.abc import Callable
import math
from typing import Optional

import e3nn_jax
import jax
import jax.numpy as jnp

import tensorial


class RadialBasis(tensorial.Attr):
    """A set of radial basis functions"""

    _number: int
    _domain: tuple[float, float]

    def __init__(self, number: int, domain=(0.0, jnp.inf)):
        self._number = number
        self._domain = domain
        super().__init__(number * e3nn_jax.Irrep(0, p=1))

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
        return self._domain

    @abc.abstractmethod
    def evaluate(self, radius):
        """Evaluate the radial basis at `r`"""

    def create_tensor(self, value) -> jnp.array:
        return super().create_tensor(self.evaluate(value))  # pylint: disable=not-callable


class E3nnRadial(RadialBasis):
    """Select a radial function from the one-hot linspace built into e3nn-jax

    see: https://e3nn-jax.readthedocs.io/en/latest/api/radial.html"""

    _basis: Callable[[float], jnp.array]
    _cutoff: float

    def __init__(self, basis: str, max_radius: float, number: int, *, cutoff=None, min_radius=0.0):
        super().__init__(number, domain=(min_radius, max_radius))
        self._basis = basis
        self._cutoff = cutoff

    @property
    def basis(self) -> str:
        return self._basis

    @property
    def cutoff(self) -> Optional[bool]:
        return self._cutoff

    def evaluate(self, radius):
        return e3nn_jax.soft_one_hot_linspace(
            radius,
            start=self.domain[0],
            end=self.domain[1],
            number=self.number,
            basis=self.basis,
            cutoff=self.cutoff,
        )


class E3nnPolyEnvelope(RadialBasis):
    """
    Polynomial envelope that can be used to make a radial basis smoothly approach zero at the cutoff
    """

    _radials: RadialBasis
    _smoothing_start: float
    _smoothing_width: float
    _envelope: Callable[[float], jax.Array]

    def __init__(self, basis: RadialBasis, smoothing_start: float, n0: int, n1: int):
        super().__init__(basis.number, basis.domain)

        if smoothing_start < basis.domain[0] or smoothing_start > basis.domain[1]:
            raise ValueError(
                f"The start of the smoothing envelope ({smoothing_start}) must be within the "
                f"domain ({basis.domain}) of the radial basis"
            )

        self._radials = basis
        self._smoothing_start = smoothing_start
        self._smoothing_width = basis.domain[1] - smoothing_start
        self._envelope = e3nn_jax.poly_envelope(n0, n1, self._smoothing_width)

    def evaluate(self, radius):
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
    radial_samples: jax.Array
    radial_step: jax.Array
    area_samples: jax.Array
    f_samples: jax.Array

    def __init__(self, radials: RadialBasis, n_samples: int):
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

    def evaluate(self, radius):
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
        return jnp.trapezoid(val_a * val_b * self.area_samples, self.radial_samples)

    def norm(self, val):
        return jnp.sqrt(self.inner_product(val, val))
