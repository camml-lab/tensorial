from collections.abc import Iterable

import beartype
import e3nn_jax as e3j
from flax import linen
import jax
import jax.numpy as jnp
import jaxtyping as jt
from jaxtyping import Array, DTypeLike, Float

from tensorial.typing import IndexArray, IntoIrreps, IrrepLike, IrrepsArrayShape

A025582 = [0, 1, 3, 7, 12, 20, 30, 44, 65, 80, 96, 122, 147, 181, 203, 251, 289]


@jt.jaxtyped(typechecker=beartype.beartype)
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
    param_dtype: DTypeLike = jnp.float32

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
        per_order: bool = False,
    ) -> (
        IrrepsArrayShape["n features irreps_out"]
        | dict[int, IrrepsArrayShape["n features irreps_out"]]
    ):
        contract = self._contract_per_order if per_order else self._contract

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

    @jt.jaxtyped(typechecker=beartype.beartype)
    def _contract_per_order(
        self,
        inputs: IrrepsArrayShape["n_feats irreps_in"],
        input_type: IndexArray[""] | None,
    ) -> dict[int, IrrepsArrayShape["n_feats irreps_out"]]:
        # pylint: disable=too-many-branches
        outputs_by_order: dict[int, dict[e3j.Irrep, Array]] = {
            o: {} for o in range(1, self.correlation_order + 1)
        }
        seen_irreps = set()

        for order in range(self.correlation_order, 0, -1):
            if self.off_diagonal:
                inp = jnp.roll(inputs.array, A025582[order - 1], axis=0)
            else:
                inp = inputs.array

            if self.symmetric_tensor_product_basis:
                basis = e3j.reduced_symmetric_tensor_product_basis(
                    inputs.irreps, order, keep_ir=self._keep_irrep_out
                )
            else:
                basis = e3j.reduced_tensor_product_basis(
                    [inputs.irreps] * order, keep_ir=self._keep_irrep_out
                )

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

                if input_type is not None:
                    weights = weights[input_type]
                else:
                    if weights.shape[0] != 1:
                        raise ValueError(
                            f"input_type=None requires num_types==1, got {weights.shape[0]}"
                        )
                    weights = weights[0]

                weights = weights * (mul**-0.5) ** self._gradient_normalisation

                if ir_out not in seen_irreps:
                    seen_irreps.add(ir_out)
                    outputs_by_order[order][ir_out] = (
                        "special",
                        jnp.einsum("...jki,kc,cj->c...i", basis_fn, weights, inp),
                    )
                else:
                    outputs_by_order[order][ir_out] = jnp.einsum(
                        "...ki,kc->c...i", basis_fn, weights
                    )

            for o in range(self.correlation_order, order - 1, -1):
                for ir_out, val in outputs_by_order[o].items():
                    if isinstance(val, tuple):
                        outputs_by_order[o][ir_out] = val[1]
                        continue

                    outputs_by_order[o][ir_out] = jnp.einsum(
                        "c...ji,cj->c...i", outputs_by_order[o][ir_out], inp
                    )

        all_irreps = set()
        for o in range(1, self.correlation_order + 1):
            all_irreps.update(outputs_by_order[o].keys())

        irreps_out = e3j.Irreps(sorted(all_irreps))

        results = {}
        for o in range(1, self.correlation_order + 1):
            chunks = []
            for _, ir in irreps_out:
                if ir in outputs_by_order[o]:
                    chunks.append(outputs_by_order[o][ir][:, None, :])
                else:
                    chunks.append(jnp.zeros((inputs.shape[0], 1, ir.dim), dtype=inputs.dtype))

            results[o] = e3j.from_chunks(irreps_out, chunks, (inputs.shape[0],))

        return results


@jt.jaxtyped(typechecker=beartype.beartype)
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
        return e3j.flax.Linear(self._target_irreps, force_irreps_out=True)(features)


@jt.jaxtyped(typechecker=beartype.beartype)
class _OrderedSymmetricTerm(linen.Module):
    """Weighted symmetric tensor contraction at a single fixed order.

    Unlike :class:`SymmetricContraction`, this module computes only the contribution
    from ``x^order``. It never accumulates across lower orders, so its parameters are
    independent of any other term. This is what makes the joint ``(a, b)`` expansion
    truly independent per pair.
    """

    order: int
    num_types: int = 1
    keep_irrep_out: str | Iterable[IrrepLike] | None = None
    gradient_normalisation: float = 0.0
    symmetric_tensor_product_basis: bool = True
    param_dtype: DTypeLike = jnp.float32

    def setup(self):
        """Initialise the keep_irrep_out set for filtering tensor product outputs."""
        # pylint: disable=attribute-defined-outside-init
        kio = self.keep_irrep_out
        if kio is None:
            self._keep_irrep_out = None
        elif isinstance(kio, str):
            self._keep_irrep_out = {e3j.Irrep(ir) for ir in e3j.Irreps(kio)}
        else:
            self._keep_irrep_out = {e3j.Irrep(ir) for ir in kio}

    @linen.compact
    @jt.jaxtyped(typechecker=beartype.beartype)
    def __call__(
        self,
        inputs: IrrepsArrayShape["... n_feats irreps_in"],
        input_type: IndexArray["..."] | None = None,
    ) -> IrrepsArrayShape["... n_feats irreps_out"]:
        """Apply the weighted symmetric tensor contraction at a single fixed order.

        This computes only the contribution from ``x^order`` without accumulating
        across lower orders, keeping parameters independent per term in the joint
        (a, b) expansion.

        Args:
            inputs: The input irreps arrays with shape ``(..., n_feats, irreps_in)``.
            input_type: Optional per-element type indices. If provided, parameters are
                indexed accordingly; otherwise requires ``num_types == 1``.

        Returns:
            The output irreps arrays with shape ``(..., n_feats, irreps_out)``.
        """
        contract = self._contract
        if input_type is not None:
            shape = jnp.broadcast_shapes(inputs.shape[:-2], input_type.shape)
            inputs = inputs.broadcast_to(shape + inputs.shape[-2:])
            input_type = jnp.broadcast_to(input_type, shape)
        for _ in range(inputs.ndim - 2):
            contract = jax.vmap(contract)
        return contract(inputs, input_type)

    def _contract(self, inputs, input_type):
        """Compute the symmetric tensor product contraction for a single order.

        Computes w · x^order where w are learnable weights and x are the inputs,
        using reduced symmetric/non-symmetric tensor product basis.

        Args:
            inputs: The input irreps array.
            input_type: Optional type indices for parameter selection.

        Returns:
            The contracted output as an IrrepsArray.
        """
        if self.symmetric_tensor_product_basis:
            basis = e3j.reduced_symmetric_tensor_product_basis(
                inputs.irreps, self.order, keep_ir=self._keep_irrep_out
            )
        else:
            basis = e3j.reduced_tensor_product_basis(
                [inputs.irreps] * self.order, keep_ir=self._keep_irrep_out
            )

        inp = inputs.array
        outputs: dict[e3j.Irrep, Array] = {}

        for (mul, ir_out), basis_fn in zip(basis.irreps, basis.chunks):
            basis_fn = basis_fn.astype(inp.dtype)

            weights = self.param(
                f"w_{ir_out}",
                linen.initializers.normal(
                    stddev=(mul**-0.5) ** (1.0 - self.gradient_normalisation)
                ),
                (self.num_types, mul, inputs.shape[0]),
                self.param_dtype,
            )
            if input_type is not None:
                weights = weights[input_type]
            else:
                if weights.shape[0] != 1:
                    raise ValueError(
                        f"input_type=None requires num_types==1, got {weights.shape[0]}"
                    )
                weights = weights[0]
            weights = weights * (mul**-0.5) ** self.gradient_normalisation

            # Special einsum (see SymmetricContraction._contract), then the remaining
            # (order - 1) multiplications. Result: w · x^order.
            value = jnp.einsum("...jki,kc,cj->c...i", basis_fn, weights, inp)
            for _ in range(self.order - 1):
                value = jnp.einsum("c...ji,cj->c...i", value, inp)

            outputs[ir_out] = value

        irreps_out = e3j.Irreps(sorted(outputs.keys()))
        return e3j.from_chunks(
            irreps_out,
            [outputs[ir][:, None, :] for (_, ir) in irreps_out],
            (inputs.shape[0],),
        )


@jt.jaxtyped(typechecker=beartype.beartype)
class JointProductBasisBlock(linen.Module):
    """Joint body-order expansion over node features and global features.

    Produces

        Σ_{(a, b) ≠ (0, 0)} Linear_{a, b}( x^a ⊗ g^b )

    for ``1 ≤ a ≤ node_correlation_order`` and ``0 ≤ b ≤ global_correlation_order``,
    including the pure-node ``(a, 0)`` and pure-global ``(0, b)`` slices. Every ``(a, b)``
    term has fully independent parameters from every other term.
    """

    irreps_out: IntoIrreps
    node_correlation_order: int
    global_correlation_order: int
    num_types: int = 1
    symmetric_tensor_product_basis: bool = True
    gradient_normalisation: str | float | None = "element"
    param_dtype: DTypeLike = jnp.float32

    def setup(self):
        # pylint: disable=attribute-defined-outside-init
        target_irreps = e3j.Irreps(self.irreps_out)
        self._target_irreps = target_irreps

        gn = self.gradient_normalisation
        if gn is None:
            gn = e3j.config("gradient_normalization")
        if isinstance(gn, str):
            gn = {"element": 0.0, "path": 1.0}[gn]

        pairs = [
            (a, b)
            for a in range(self.node_correlation_order + 1)
            for b in range(self.global_correlation_order + 1)
            if not (a == 0 and b == 0)
        ]
        if not pairs:
            raise ValueError(
                "JointProductBasisBlock requires at least one non-(0, 0) term; "
                "got node_correlation_order=0 and global_correlation_order=0"
            )
        self._pairs = pairs

        x_terms: dict = {}
        g_terms: dict = {}
        output_linears: dict = {}

        for a, b in pairs:
            if a > 0:
                x_terms[(a, b)] = _OrderedSymmetricTerm(
                    order=a,
                    num_types=self.num_types,
                    keep_irrep_out=None,  # no filtering; filter after tensor product
                    gradient_normalisation=gn,
                    symmetric_tensor_product_basis=self.symmetric_tensor_product_basis,
                    param_dtype=self.param_dtype,
                    name=f"x_{a}_{b}",
                )
            if b > 0:
                g_terms[(a, b)] = _OrderedSymmetricTerm(
                    order=b,
                    num_types=1,  # one global feature vector per graph
                    keep_irrep_out=None,
                    gradient_normalisation=gn,
                    symmetric_tensor_product_basis=self.symmetric_tensor_product_basis,
                    param_dtype=self.param_dtype,
                    name=f"g_{a}_{b}",
                )

            output_linears[(a, b)] = e3j.flax.Linear(
                target_irreps,
                num_indexed_weights=self.num_types if self.num_types > 1 else None,
                name=f"out_{a}_{b}",
                force_irreps_out=True,
            )

        self._x_terms = x_terms
        self._g_terms = g_terms
        self._output_linears = output_linears

    @linen.compact
    @jt.jaxtyped(typechecker=beartype.beartype)
    def __call__(
        self,
        x: IrrepsArrayShape["n_node n_featsXnode_irreps"],
        g: IrrepsArrayShape["n_node n_featsXglobal_irreps"],
        input_type: IndexArray["n_node"] | None = None,
    ) -> IrrepsArrayShape["n_node irreps_out"]:
        x_in = x.mul_to_axis().remove_zero_chunks()
        g_in = g.mul_to_axis().remove_zero_chunks()

        out = None
        for a, b in self._pairs:
            if a > 0 and b > 0:
                x_term = self._x_terms[(a, b)](x_in, input_type)
                g_term = self._g_terms[(a, b)](g_in, None)
                joint = e3j.tensor_product(x_term, g_term, filter_ir_out=self._target_irreps)
            elif a > 0:
                joint = self._x_terms[(a, b)](x_in, input_type)
            else:
                joint = self._g_terms[(a, b)](g_in, None)

            joint = joint.axis_to_mul()

            if self.num_types > 1:
                if input_type is None:
                    raise ValueError("input_type must be provided when num_types > 1")
                term = self._output_linears[(a, b)](input_type, joint)
            else:
                term = self._output_linears[(a, b)](joint)

            out = term if out is None else out + term

        return out

    @linen.compact
    def _debug_intermediates(self, x, g, input_type=None):
        """Diagnostic only: pre-linear intermediates per (a, b)."""
        x_in = x.mul_to_axis().remove_zero_chunks()
        g_in = g.mul_to_axis().remove_zero_chunks()

        result = {}
        for a, b in self._pairs:
            if a > 0 and b > 0:
                xt = self._x_terms[(a, b)](x_in, input_type)
                gt = self._g_terms[(a, b)](g_in, None)
                joint = e3j.tensor_product(xt, gt, filter_ir_out=self._target_irreps)
            elif a > 0:
                joint = self._x_terms[(a, b)](x_in, input_type)
            else:
                joint = self._g_terms[(a, b)](g_in, None)
            result[(a, b)] = joint.axis_to_mul()
        return result
