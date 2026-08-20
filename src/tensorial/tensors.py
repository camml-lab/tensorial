from typing import Literal

import beartype
import e3nn_jax as e3j
import jax
import jax.numpy as jnp
import jaxtyping as jt
from jaxtyping import Array, Float, Int

from tensorial.typing import IrrepsArrayShape

from . import base

__all__ = "SphericalHarmonic", "CartesianTensor", "NoOp", "OneHot", "AsIrreps"


class NoOp(base.Attr[e3j.IrrepsArray]):
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


class AsIrreps(base.Attr[jt.ArrayLike]):
    def _validate(self, value):
        assert isinstance(value, jnp.ndarray), "Expected a jnp.ndarray"
        assert value.shape[-1] == self.irreps.dim, "Dimension mismatch"

    @jt.jaxtyped(typechecker=beartype.beartype)
    def create_tensor(self, value: jt.ArrayLike) -> e3j.IrrepsArray:
        self._validate(value)
        return e3j.IrrepsArray(self.irreps, value)

    @jt.jaxtyped(typechecker=beartype.beartype)
    def from_tensor(self, tensor: e3j.IrrepsArray) -> e3j.IrrepsArray:
        assert tensor.irreps == self.irreps, "Irreps mismatch"
        return tensor


class SphericalHarmonic(base.Attr[jax.Array | e3j.IrrepsArray]):
    """An attribute that is the spherical harmonics evaluated as some values"""

    normalise: bool
    normalisation: Literal["integral", "component", "norm"] | None = None
    algorithm: tuple[str] | None = None

    def __init__(
        self,
        irreps,
        normalise,
        normalisation: Literal["integral", "component", "norm"] | None = None,
        *,
        algorithm: tuple[str] = None,
    ):
        super().__init__(irreps)
        self.normalise = normalise
        self.normalisation = normalisation
        self.algorithm = algorithm

    def create_tensor(self, value: jax.Array | e3j.IrrepsArray) -> jnp.array:
        return e3j.spherical_harmonics(
            self.irreps,
            base.as_array(value),
            normalize=self.normalise,
            normalization=self.normalisation,
            algorithm=self.algorithm,
        )


class OneHot(base.Attr[Int[Array, "n_vals"]]):
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
    def create_tensor(self, value: Int[Array, "n_vals"]) -> IrrepsArrayShape["n_node num_classes"]:
        return e3j.IrrepsArray(self.irreps, jax.nn.one_hot(value, self.num_classes))


class CartesianTensor(base.Attr[jt.ArrayLike]):
    formula: str
    keep_ir: e3j.Irreps | list[e3j.Irrep] | None
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
    def create_tensor(self, value: jt.ArrayLike) -> e3j.IrrepsArray:
        # Was hardcoded to the 2-index case ("ij,ijz->z"), which silently broke for any
        # formula with a different number of indices (e.g. the 3-index "ijk=jik" used for
        # rank-3 tensors like Pockels). `self._indices` already holds the right subscript
        # letters for the formula ("ij", "ijk", ...) so we just plug it into the einsum,
        # with a leading "..." to also support a batch dimension.
        return super().create_tensor(  # pylint: disable=not-callable
            jnp.einsum(f"...{self._indices},{self._indices}z->...z", value, self.change_of_basis)
        )

    @jt.jaxtyped(typechecker=beartype.beartype)
    def from_tensor(
        self,
        tensor: IrrepsArrayShape["irreps"] | IrrepsArrayShape["batch irreps"],
    ) -> Float[jax.Array, "..."] | Float[jax.Array, "batch ..."]:
        """Take an irrep tensor and perform the change of basis transformation back to a Cartesian
        tensor

        Args:
            tensor: the irrep tensor

        Returns:
            the Cartesian tensor
        """
        rot = self.change_of_basis.reshape(-1, self.change_of_basis.shape[-1])
        cartesian = base.as_array(tensor) @ rot.T
        return cartesian.reshape((*tensor.shape[:-1], *self.change_of_basis.shape[:-1]))
