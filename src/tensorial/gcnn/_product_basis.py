from collections.abc import Iterable

import beartype
import e3nn_jax as e3j
from flax import linen
import jax
import jax.numpy as jnp
import jaxtyping as jt
from jaxtyping import Array, Float

from tensorial.typing import IndexArray, IrrepLike, IrrepsArrayShape

A025582 = [0, 1, 3, 7, 12, 20, 30, 44, 65, 80, 96, 122, 147, 181, 203, 251, 289]


class SymmetricContraction(linen.Module):
    """Symmetric tensor contraction up to a given correlation order.

    Based on implementation from:

        https://github.com/ACEsuit/mace-jax/blob/main/mace_jax/modules/symmetric_contraction.py
    """

    correlation_order: int
    keep_irrep_out: str | Iterable[IrrepLike]

    num_types: int = 1
    gradient_normalisation: str | float | None = None
    symmetric_tensor_product_basis: bool = True
    off_diagonal: bool = False
    param_dtype = jnp.float32

    def setup(self):
        # pylint: disable=attribute-defined-outside-init
        # Gradient normalisation
        gradient_normalisation = self.gradient_normalisation
        if gradient_normalisation is None:
            gradient_normalisation = e3j.config("gradient_normalization")
        if isinstance(gradient_normalisation, str):
            gradient_normalisation = {"element": 0.0, "path": 1.0}[gradient_normalisation]
        self._gradient_normalisation = gradient_normalisation

        # Output irreps to keep
        keep_irrep_out = self.keep_irrep_out
        if isinstance(self.keep_irrep_out, str):
            keep_irrep_out = e3j.Irreps(self.keep_irrep_out)
            assert all(mul == 1 for mul, _ in keep_irrep_out)

        self._keep_irrep_out = {e3j.Irrep(ir) for ir in keep_irrep_out}

    @linen.compact
    @jt.jaxtyped(typechecker=beartype.beartype)
    def __call__(
        self,
        inputs: IrrepsArrayShape["n features irreps"],
        input_type: IndexArray["n"] | None,
    ) -> IrrepsArrayShape["n features irreps_out"]:
        contract = self._contract

        if input_type is not None:
            # Align batch shape between inputs and input_type
            shape = jnp.broadcast_shapes(inputs.shape[:-2], input_type.shape)
            inputs = inputs.broadcast_to(shape + inputs.shape[-2:])
            input_type = jnp.broadcast_to(input_type, shape)

        for _ in range(inputs.ndim - 2):
            contract = jax.vmap(contract)

        return contract(inputs, input_type)

    @jt.jaxtyped(typechecker=beartype.beartype)
    def _contract(
        self,
        inputs: IrrepsArrayShape["n_feats irreps_in"],
        input_type: IndexArray[""] | None,
    ) -> IrrepsArrayShape["n_feats irreps_out"]:
        """This operation is parallel on the feature dimension (but each feature has its own
        parameters)
        Efficient implementation of:

            vmap(lambda w, x: FunctionalLinear(irreps_out)(
                w, concatenate([x, tensor_product(x, x), tensor_product(x, x, x), ...])))(w, x)

        up to x power ``self.correlation_order``

        Args:
            inputs: the contraction inputs
            input_type: per-element type index, or None to share one set of
                parameters across all inputs (requires ``num_types == 1``).

        Returns:
            the contraction outputs
        """
        outputs: dict[e3j.Irrep, Array] = dict()
        for order in range(self.correlation_order, 0, -1):  # correlation_order, ..., 1
            if self.off_diagonal:
                inp = jnp.roll(inputs.array, A025582[order - 1], axis=0)
            else:
                inp = inputs.array

            # Create the basis
            if self.symmetric_tensor_product_basis:
                basis = e3j.reduced_symmetric_tensor_product_basis(
                    inputs.irreps, order, keep_ir=self._keep_irrep_out
                )
            else:
                basis = e3j.reduced_tensor_product_basis(
                    [inputs.irreps] * order, keep_ir=self._keep_irrep_out
                )

            # ((w3 x + w2) x + w1) x
            #  \-----------/
            #       out

            for (mul, ir_out), basis_fn in zip(basis.irreps, basis.chunks):
                basis_fn: Float[Array, "irreps_in^order multiplicity irreps_out"] = basis_fn.astype(
                    inp.dtype
                )

                weights: Float[Array, "multiplicity n_feats"] = self.param(
                    f"w{order}_{ir_out}",
                    linen.initializers.normal(
                        stddev=(mul**-0.5) ** (1.0 - self._gradient_normalisation)
                    ),
                    (self.num_types, mul, inputs.shape[0]),
                    self.param_dtype,
                )

                # Index by type (or squeeze the type axis when there is no type index)
                if input_type is not None:
                    weights = weights[input_type]
                else:
                    # We only ever share weights across a single "type" here; if you
                    # want num_types > 1 with input_type=None, decide how to combine
                    # them (mean? sum?) before this line.
                    if weights.shape[0] != 1:
                        raise ValueError(
                            f"input_type=None requires num_types==1, got {weights.shape[0]}"
                        )
                    weights = weights[0]

                # normalize the weights
                weights = weights * (mul**-0.5) ** self._gradient_normalisation

                if ir_out not in outputs:
                    outputs[ir_out] = (
                        "special",
                        jnp.einsum("...jki,kc,cj->c...i", basis_fn, weights, inp),
                    )  # [n_feats, (irreps_x.dim)^(oder-1), ir_out.dim]
                else:
                    outputs[ir_out] += jnp.einsum(
                        "...ki,kc->c...i", basis_fn, weights
                    )  # [n_feats, (irreps_x.dim)^order, ir_out.dim]

            # ((w3 x + w2) x + w1) x
            #  \----------------/
            #         out (in the normal case)

            for ir_out, val in outputs.items():
                if isinstance(val, tuple):
                    outputs[ir_out] = val[1]
                    continue  # already done (special case optimisation above)

                value: Float[Array, "n_feats irreps_in^(oder-1) irreps_out"] = jnp.einsum(
                    "c...ji,cj->c...i", outputs[ir_out], inp
                )
                outputs[ir_out] = value

            # ((w3 x + w2) x + w1) x
            #  \-------------------/
            #           out

        irreps_out = e3j.Irreps(sorted(outputs.keys()))
        output: IrrepsArrayShape["n_feats irreps_out"] = e3j.from_chunks(
            irreps_out,
            [outputs[ir][:, None, :] for (_, ir) in irreps_out],
            (inputs.shape[0],),
        )
        return output


class EquivariantProductBasisBlock(linen.Module):
    irreps_out: e3j.Irreps
    correlation_order: int
    num_types: int
    symmetric_tensor_product_basis: bool = True
    off_diagonal: bool = False

    def setup(self):
        # pylint: disable=attribute-defined-outside-init
        self._target_irreps = e3j.Irreps(self.irreps_out)
        self.symmetric_contractions = SymmetricContraction(
            keep_irrep_out={ir for _, ir in e3j.Irreps(self._target_irreps)},
            correlation_order=self.correlation_order,
            num_types=self.num_types,
            gradient_normalisation="element",  # NOTE: This is to copy mace-torch
            symmetric_tensor_product_basis=self.symmetric_tensor_product_basis,
            off_diagonal=self.off_diagonal,
        )

    @linen.compact
    @jt.jaxtyped(typechecker=beartype.beartype)
    def __call__(
        self,
        features: IrrepsArrayShape["n n_featsXirreps"],
        input_type: IndexArray["n"] | None = None,
    ) -> IrrepsArrayShape["n irreps_out"]:
        features = features.mul_to_axis().remove_zero_chunks()
        features = self.symmetric_contractions(features, input_type)
        features = features.axis_to_mul()
        return e3j.flax.Linear(self._target_irreps)(features)
