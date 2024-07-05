import abc

import jax
import jax.numpy as jnp


class Function:
    """Base class for functions"""

    def __add__(self, other) -> "Sum":
        return Sum((self, other))

    def __call__(self, x):
        return self.evaluate(x)

    @abc.abstractmethod
    def evaluate(self, x):
        """Evaluate the function at point `x`"""


class DiracDelta(Function):
    """A Dirac delta with an optional weight"""

    def __init__(self, pos, weight=1.0):
        self.pos = pos
        self.weight = weight

    def evaluate(self, x):
        return jax.lax.cond(not (self.pos - x).any(), lambda: self.weight, lambda: 0.0)


class IsotropicGaussian(Function):
    """A 3D Gaussian with an optional weight and scalar sigma"""

    def __init__(self, pos, sigma, weight=1.0) -> None:
        super().__init__()
        self.pos = pos
        self.sigma = sigma
        self.weight = weight

    def evaluate(self, x):
        return (
            self.weight
            / (jnp.sqrt(2 * jnp.pi))
            * jnp.exp(-(jnp.sum((x - self.pos) ** 2)) / (2 * self.sigma**2))
        )


class Sum(Function):
    """A sum of other functions"""

    def __init__(self, functions: tuple) -> None:
        super().__init__()
        transformed = []
        for func in functions:
            if isinstance(func, Sum):
                transformed.extend(func.functions)
            else:
                transformed.append(func)
        self.functions = tuple(transformed)

    def evaluate(self, x):
        return jnp.sum(func(x) for func in self.functions)
