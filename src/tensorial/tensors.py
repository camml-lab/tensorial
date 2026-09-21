"""Built-in :class:`tensorial.base.Attr` implementations.

This module provides a small collection of ready-made attributes that map raw
values into :class:`e3nn_jax.IrrepsArray` tensors.  Each class corresponds to a
common tensorial quantity (spherical harmonics, one-hot encodings, cartesian
tensors, ...) and pairs naturally with the generic machinery in
:mod:`tensorial.base`.
"""

from typing import Literal

import beartype
import e3nn_jax as e3j
import jax
import jax.numpy as jnp
import jaxtyping as jt
from jaxtyping import Array, Float, Int
import numpy as np

from tensorial.typing import IrrepsArrayShape

from . import base, nn_utils

__all__ = "SphericalHarmonic", "CartesianTensor", "NoOp", "OneHot", "AsIrreps"


class NoOp(base.Attr[e3j.IrrepsArray]):
    """An attribute that keeps IrrepsArrays with specified irreps unchanged.

    The raw value is expected to *already* be an :class:`e3nn_jax.IrrepsArray`
    with exactly the irreps this attribute declares; both directions simply
    validate and pass the value through.
    """

    def _validate(self, value):
        assert isinstance(value, e3j.IrrepsArray), "Expected an IrrepsArray"
        assert value.irreps == self.irreps, "Irreps mismatch"

    def create_tensor(self, value: e3j.IrrepsArray) -> e3j.IrrepsArray:
        """Return the input (already an IrrepsArray) unchanged."""
        self._validate(value)
        return value

    def from_tensor(self, tensor: e3j.IrrepsArray) -> e3j.IrrepsArray:
        """Return the tensor unchanged."""
        self._validate(tensor)
        return tensor


class AsIrreps(base.Attr[jt.ArrayLike]):
    """Wrap a plain array into an :class:`e3nn_jax.IrrepsArray` with fixed irreps.

    The incoming value must be an array whose final axis matches the total
    dimension of :attr:`irreps`.  Unlike :class:`NoOp` the input does not
    need to already be an :class:`e3nn_jax.IrrepsArray`; it is interpreted
    purely by shape.
    """

    def _validate(self, value):
        assert isinstance(value, jnp.ndarray), "Expected a jnp.ndarray"
        assert value.shape[-1] == self.irreps.dim, "Dimension mismatch"

    @jt.jaxtyped(typechecker=beartype.beartype)
    def create_tensor(self, value: jt.ArrayLike) -> e3j.IrrepsArray:
        """Wrap the input array in an :class:`e3nn_jax.IrrepsArray` with :attr:`irreps`."""
        self._validate(value)
        return e3j.IrrepsArray(self.irreps, value)

    @jt.jaxtyped(typechecker=beartype.beartype)
    def from_tensor(self, tensor: e3j.IrrepsArray) -> e3j.IrrepsArray:
        """Return the tensor unchanged after checking its irreps match."""
        assert tensor.irreps == self.irreps, "Irreps mismatch"
        return tensor


class SphericalHarmonic(base.Attr[jax.Array | e3j.IrrepsArray]):
    """Expand a vector/position into the given irreps using spherical harmonics.

    The raw value is expected to be a unit (or plain) vector with shape
    ``(..., 3)``; :meth:`create_tensor` evaluates the real spherical harmonics
    for :attr:`irreps` on that vector and returns the
    :class:`e3nn_jax.IrrepsArray` result.

    Args:
        irreps: the irreps to expand into
        normalise: whether the harmonics are normalised
        normalisation: type of normalisation, one of ``integral``, ``component`` or ``norm``
        algorithm: optional algorithm hint for :func:`e3nn_jax.spherical_harmonics`
    """

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
        """Build the tensor by evaluating spherical harmonics of the input vector(s)."""
        return e3j.spherical_harmonics(
            self.irreps,
            base.as_array(value),
            normalize=self.normalise,
            normalization=self.normalisation,
            algorithm=self.algorithm,
        )


class OneHot(base.Attr[Int[Array, "n_vals 1"]]):
    """One-hot encoding of a class index as a direct sum of even scalars.

    The attribute's irreps are ``num_classes * o(0)``, i.e. one scalar
    multiplicity per class.  This makes one-hot vectors natively a
    :class:`e3nn_jax.IrrepsArray`, which they then travel around with.

    Args:
        num_classes: number of possible classes.  Either this or ``types`` must be given.
        types: explicit list of type labels (used to build the index mapping when
            the raw values are actual type ids, not consecutive integers).

    Raises:
        ValueError: if neither ``num_classes`` nor ``types`` is provided.
    """

    _types: np.ndarray

    def __init__(self, num_classes: int = None, types: list[int] = None):
        if num_classes is None:
            if types is None:
                raise ValueError(
                    "Need to specify the number of one hot classes, or the list of types, "
                    "got neither."
                )
            num_classes = len(types)
            types = np.array(types)
        else:
            types = np.arange(num_classes)

        super().__init__(num_classes * e3j.Irrep(0, 1))
        self._types = types

    @property
    def num_classes(self) -> int:
        """The number of one-hot classes, read back from :attr:`irreps`."""
        mul_irrep = self.irreps[0]
        if isinstance(mul_irrep, e3j.MulIrrep):
            return mul_irrep.mul
        raise ValueError("Expected self.irreps to contain a MulIrrep.")

    @jt.jaxtyped(typechecker=beartype.beartype)
    def create_tensor(
        self, value: Int[Array, "n_vals 1"]
    ) -> IrrepsArrayShape["n_node num_classes"]:
        """One-hot encode the class indices into the scalar irrep basis.

        Args:
            value: An array of integer class labels with shape ``(n_vals, 1)``.

        Returns:
            An `e3j.IrrepsArray` of shape ``(n_node, num_classes)`` with the
            one-hot encoding of each value.
        """
        sequential = nn_utils.vwhere(value[:, 0], self._types)
        return e3j.IrrepsArray(self.irreps, jax.nn.one_hot(sequential, self.num_classes))


class CartesianTensor(base.Attr[jt.ArrayLike]):
    """Convert a Cartesian tensor of a given rank into an :class:`e3nn_jax.IrrepsArray`.

    The raw value's last axis is interpreted as the Cartesian tensor's size
    (e.g. 9 for rank 2, 27 for rank 3); :meth:`create_tensor` applies the
    change-of-basis matrix of the *reduced tensor product* basis so the value
    lives in the irrep basis instead.  :meth:`from_tensor` performs the inverse
    change of basis back to Cartesian components.

    Args:
        formula: an einsum-style formula describing the Cartesian tensor, e.g.
            ``"zz=xy"`` for a rank-2 tensor (first character of the left-hand
            side determines the rank).
        keep_ir: optional selection of which irreps to keep in the reduced basis
        **irreps_dict: additional irreps arguments forwarded to
            :func:`e3nn_jax.reduced_tensor_product_basis`
    """

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
        """Project the input onto the cartesian tensor basis via the change-of-basis.

        The value is multiplied by the stored change-of-basis matrix with an
        einsum over the cartesian indices, then delegated to the base class.
        """
        # Construct the einsum string dynamically based on the rank
        indices = self._indices
        einsum_str = f"{indices},{indices}z->z"
        return super().create_tensor(  # pylint: disable=not-callable
            jnp.einsum(einsum_str, value, self.change_of_basis)
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
