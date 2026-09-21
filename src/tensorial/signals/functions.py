"""Abstract signal values (Dirac deltas, Gaussians, sums)."""

import abc

import jax
import jax.numpy as jnp


class Function:
    """Base class for functions."""

    def __add__(self, other) -> "Sum":
        """Return the :class:`Sum` of ``self`` and ``other``."""
        return Sum((self, other))

    def __call__(self, x):
        """Call the function at point ``x`` (shorthand for :meth:`evaluate`)."""
        return self.evaluate(x)

    @abc.abstractmethod
    def evaluate(self, x):
        """Evaluate the function at point ``x``."""


class DiracDelta(Function):
    """A Dirac delta distribution with an optional weight."""

    def __init__(self, pos, weight=1.0):
        """Position and weight of the delta.

        Args:
            pos: the location of the delta (a vector of shape ``(3,)``).
            weight: amplitude of the delta (default 1.0).
        """
        self.pos = pos
        self.weight = weight

    def evaluate(self, x):
        """Return the weight at the delta position and zero elsewhere."""
        return jax.lax.cond(not (self.pos - x).any(), lambda: self.weight, lambda: 0.0)


class IsotropicGaussian(Function):
    """A 3D Gaussian with an optional weight and scalar sigma."""

    def __init__(self, pos, sigma, weight=1.0) -> None:
        """Construct a Gaussian centred at ``pos``.

        Args:
            pos: the centroid (a vector of shape ``(3,)``).
            sigma: the standard deviation.
            weight: the peak value at the centroid (default 1.0).
        """
        super().__init__()
        self.pos = pos
        self.sigma = sigma
        self.weight = weight

    def evaluate(self, x):
        """Evaluate the Gaussian at point ``x``."""
        return (
            self.weight
            / (jnp.sqrt(2 * jnp.pi))
            * jnp.exp(-(jnp.sum((x - self.pos) ** 2)) / (2 * self.sigma**2))
        )


class Sum(Function):
    """A sum of other functions."""

    def __init__(self, functions: tuple) -> None:
        """Build a sum, flattening :class:`Sum` entries into a single tuple.

        Args:
            functions: the tuple of functions to sum.
        """
        super().__init__()
        transformed = []
        for func in functions:
            if isinstance(func, Sum):
                transformed.extend(func.functions)
            else:
                transformed.append(func)
        self.functions = tuple(transformed)

    def evaluate(self, x):
        """Sum each member function's value at point ``x``."""
        return jnp.sum(func(x) for func in self.functions)
