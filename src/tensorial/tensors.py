from typing import Literal, Optional, Union

import beartype
import e3nn_jax as e3j
import jax
import jax.numpy as jnp
import jaxtyping as jt

from . import base, typing

__all__ = "SphericalHarmonic", "CartesianTensor", "NoOp", "OneHot", "AsIrreps"


class NoOp(base.Attr):
    """An attribute that keeps IrrepsArrays with specified irreps unchanged"""

    def _validate(self, value):
        assert isinstance(value, e3j.IrrepsArray), "Expected an IrrepsArray"
        assert value.irreps == self.irreps, "Irreps mismatch"

    def create_tensor(self, value: e3j.IrrepsArray) -> e3j.IrrepsArray:
        self._validate(value)
        return value

    def from_tensor(self, tensor: e3j.IrrepsArray) -> e3j.IrrepsArray:
        self._validate(tensor)
        return tensor


class AsIrreps(base.Attr):
    def _validate(self, value):
        assert isinstance(value, jnp.ndarray), "Expected a jnp.ndarray"
        assert value.shape[-1] == self.irreps.dim, "Dimension mismatch"

    @jt.jaxtyped(typechecker=beartype.beartype)
    def create_tensor(self, value: jax.typing.ArrayLike) -> e3j.IrrepsArray:
        self._validate(value)
        return e3j.IrrepsArray(self.irreps, value)

    @jt.jaxtyped(typechecker=beartype.beartype)
    def from_tensor(self, tensor: e3j.IrrepsArray) -> e3j.IrrepsArray:
        assert tensor.irreps == self.irreps, "Irreps mismatch"
        return tensor


class SphericalHarmonic(base.Attr):
    """An attribute that is the spherical harmonics evaluated as some values"""

    normalise: bool
    normalisation: Optional[Literal["integral", "component", "norm"]] = None
    algorithm: Optional[tuple[str]] = None

    def __init__(
        self,
        irreps,
        normalise,
        normalisation: Optional[Literal["integral", "component", "norm"]] = None,
        *,
        algorithm: tuple[str] = None,
    ):
        super().__init__(irreps)
        self.normalise = normalise
        self.normalisation = normalisation
        self.algorithm = algorithm

    def create_tensor(self, value: Union[jax.Array, e3j.IrrepsArray]) -> jnp.array:
        return e3j.spherical_harmonics(
            self.irreps,
            base.as_array(value),
            normalize=self.normalise,
            normalization=self.normalisation,
            algorithm=self.algorithm,
        )


class OneHot(base.Attr):
    """One-hot encoding as a direct sum of even scalars"""

    def __init__(self, num_classes: int):
        super().__init__(num_classes * e3j.Irrep(0, 1))

    @property
    def num_classes(self) -> int:
        mul_irrep = self.irreps[0]
        if isinstance(mul_irrep, e3j.MulIrrep):
            return mul_irrep.mul
        raise ValueError("Expected self.irreps to contain a MulIrrep.")

    @jt.jaxtyped(typechecker=beartype.beartype)
    def create_tensor(self, value: jt.Int[typing.ArrayType, "n_vals"]) -> e3j.IrrepsArray:
        return e3j.IrrepsArray(self.irreps, jax.nn.one_hot(value, self.num_classes))


class CartesianTensor(base.Attr):
    formula: str
    keep_ir: Optional[Union[e3j.Irreps, list[e3j.Irrep]]]
    irreps_dict: dict
    change_of_basis: jax.Array
    _indices: str

    def __init__(self, formula: str, keep_ir=None, **irreps_dict) -> None:
        self.formula = formula
        self.keep_ir = keep_ir
        self.irreps_dict = irreps_dict
        self._indices = formula.split("=")[0].replace("-", "")

        # Construct the change of basis arrays
        cob = e3j.reduced_tensor_product_basis(formula, keep_ir=self.keep_ir, **self.irreps_dict)
        self.change_of_basis = cob.array
        super().__init__(cob.irreps)

    @jt.jaxtyped(typechecker=beartype.beartype)
    def create_tensor(self, value: jax.typing.ArrayLike) -> e3j.IrrepsArray:
        return super().create_tensor(  # pylint: disable=not-callable
            jnp.einsum("ij,ijz->z", value, self.change_of_basis)
        )

    @jt.jaxtyped(typechecker=beartype.beartype)
    def from_tensor(
        self,
        tensor: Union[typing.IrrepsArrayShape["irreps"], typing.IrrepsArrayShape["batch irreps"]],
    ) -> Union[jt.Float[jax.Array, "..."], jt.Float[jax.Array, "batch ..."]]:
        """
        Take an irrep tensor and perform the change of basis transformation back to a Cartesian
        tensor

        :param tensor: the irrep tensor
        :return: the Cartesian tensor
        """
        rot = self.change_of_basis.reshape(-1, self.change_of_basis.shape[-1])
        cartesian = base.as_array(tensor) @ rot.T
        return cartesian.reshape((*tensor.shape[:-1], *self.change_of_basis.shape[:-1]))
