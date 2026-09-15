from collections.abc import Callable, Iterable
import functools
import math
from typing import Literal

import beartype
import e3nn_jax as e3j
from flax import linen
import jax
import jax.numpy as jnp
import jaxtyping as jt
from jaxtyping import Array, Bool, Float, Int
import jraph

from tensorial import gcnn, nn_utils
from tensorial.typing import IndexArray, IntoIrreps, IrrepLike, IrrepsArrayShape

from . import _base, _message_passing, experimental, keys

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
        inputs: IrrepsArrayShape["n_node features irreps"],
        input_type: IndexArray["n_node"],
    ) -> IrrepsArrayShape["n_node features irreps_out"]:
        # Treat batch indices using vmap
        shape = jnp.broadcast_shapes(inputs.shape[:-2], input_type.shape)
        inputs = inputs.broadcast_to(shape + inputs.shape[-2:])
        input_type = jnp.broadcast_to(input_type, shape)

        contract = self._contract
        for _ in range(inputs.ndim - 2):
            contract = jax.vmap(contract)

        return contract(inputs, input_type)

    @jt.jaxtyped(typechecker=beartype.beartype)
    def _contract(
        self, inputs: IrrepsArrayShape["n_feats irreps_in"], input_type: IndexArray[""]
    ) -> IrrepsArrayShape["n_feats irreps_out"]:
        """This operation is parallel on the feature dimension (but each feature has its own
        parameters)
        Efficient implementation of:

            vmap(lambda w, x: FunctionalLinear(irreps_out)(
                w, concatenate([x, tensor_product(x, x), tensor_product(x, x, x), ...])))(w, x)

        up to x power ``self.correlation_order``

        Args:
            inputs: the contraction inputs
            input_type: the contraction index

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
                # Index by type
                weights = weights[input_type]  # pylint: disable=unsubscriptable-object

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
        node_features: IrrepsArrayShape["n_node n_featsXirreps"],
        node_types: IndexArray["n_node"],
    ) -> IrrepsArrayShape["n_node irreps_out"]:
        node_features = node_features.mul_to_axis().remove_zero_chunks()
        node_features = self.symmetric_contractions(node_features, node_types)
        node_features = node_features.axis_to_mul()
        return e3j.flax.Linear(self._target_irreps)(node_features)


class InteractionBlock(linen.Module):
    irreps_out: IntoIrreps

    # Normalisation
    avg_num_neighbours: float | dict[int, float] = 1.0
    epsilon: float | None = None

    radial_activation: str | nn_utils.ActivationFunction = "swish"

    def setup(self):
        # pylint: disable=attribute-defined-outside-init
        self._message_passing = _message_passing.MessagePassingConvolution(
            self.irreps_out,
            avg_num_neighbours=self.avg_num_neighbours,
            epsilon=self.epsilon,
            radial_activation=self.radial_activation,
        )
        self._linear_down = e3j.flax.Linear(self.irreps_out, name="linear_down")

    @linen.compact
    @jt.jaxtyped(typechecker=beartype.beartype)
    def __call__(
        self,
        node_features: IrrepsArrayShape["n_node node_irreps"],
        edge_features: IrrepsArrayShape["n_edge edge_irreps"],
        radial_embedding: Float[jnp.ndarray, "n_edge radial_embeddings"],
        senders: Int[Array, "n_edge"],
        receivers: Int[Array, "n_edge"],
        *,
        node_types: Int[Array, "n_node"] | None = None,
        edge_mask: Bool[Array, "n_edge"] | None = None,
    ) -> IrrepsArrayShape["n_node target_irreps"]:
        node_features = e3j.flax.Linear(node_features.irreps, name="linear_up")(node_features)

        node_features = self._message_passing(
            node_features,
            edge_features,
            radial_embedding,
            senders,
            receivers,
            edge_mask=edge_mask,
            node_types=node_types,
        )

        node_features = self._linear_down(node_features)
        assert node_features.ndim == 2

        return node_features


class NonLinearReadoutBlock(linen.Module):
    hidden_irreps: IntoIrreps
    output_irreps: IntoIrreps
    activation: Callable | None = None
    gate: Callable | None = None

    def setup(self) -> None:
        # pylint: disable=attribute-defined-outside-init
        hidden_irreps = e3j.Irreps(self.hidden_irreps)
        output_irreps = e3j.Irreps(self.output_irreps)

        # Get multiplicity of (l > 0) irreps
        num_vectors = hidden_irreps.filter(drop=["0e", "0o"]).num_irreps
        self._linear = e3j.flax.Linear((hidden_irreps + e3j.Irreps(f"{num_vectors}x0e")).simplify())
        self._linear_out = e3j.flax.Linear(output_irreps, force_irreps_out=True)

    def __call__(
        self, inputs: IrrepsArrayShape["n_node irreps"]
    ) -> IrrepsArrayShape["n_node output_irreps"]:
        inputs = self._linear(inputs)
        inputs = e3j.gate(inputs, even_act=self.activation, even_gate_act=self.gate)
        return self._linear_out(inputs)


class MaceLayer(linen.Module):
    """A MACE layer composed of:
    * Interaction block
    * Normalisation
    * Product basis
    * (optional) self connection
    """

    irreps_out: IntoIrreps
    num_types: int

    # Interaction
    num_features: int | None
    interaction_irreps: IntoIrreps
    #   Radial
    radial_activation: Callable

    # Normalisation
    epsilon: float | None
    avg_num_neighbours: float | dict[int, float] | linen.FrozenDict[int, float]

    # Product basis
    hidden_irreps: IntoIrreps
    correlation_order: int
    symmetric_tensor_product_basis: bool
    off_diagonal: bool

    soft_normalisation: float | None
    skip_connection: bool = True

    def setup(self):
        # pylint: disable=attribute-defined-outside-init
        hidden_irreps = e3j.Irreps(self.hidden_irreps)
        interaction_irreps = e3j.Irreps(self.interaction_irreps)

        if self.num_features is None:
            num_features = functools.reduce(math.gcd, (mul for mul, _ in hidden_irreps))
            hidden_irreps = e3j.Irreps([(mul // num_features, ir) for mul, ir in hidden_irreps])
        else:
            num_features = self.num_features

        self._interaction_block = InteractionBlock(
            num_features * interaction_irreps,
            avg_num_neighbours=self.avg_num_neighbours,
            epsilon=self.epsilon,
            radial_activation=self.radial_activation,
        )
        self._product_basis = EquivariantProductBasisBlock(
            irreps_out=num_features * hidden_irreps,
            correlation_order=self.correlation_order,
            num_types=self.num_types,
            symmetric_tensor_product_basis=self.symmetric_tensor_product_basis,
            off_diagonal=self.off_diagonal,
        )
        if self.skip_connection:
            self._skip_connection = e3j.flax.Linear(
                num_features * hidden_irreps,
                num_indexed_weights=self.num_types,
                name="skip_connection",
                force_irreps_out=True,
            )
        else:
            self._skip_connection = None

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __call__(
        self,
        node_features: IrrepsArrayShape["n_node node_irreps"],
        edge_features: IrrepsArrayShape["n_edge edge_irreps"],
        node_types: Int[Array, "n_node"],
        radial_embedding: Float[Array, "n_edge radial_embedding"],
        senders: Int[Array, "n_edge"],
        receivers: Int[Array, "n_edge"],
        *,
        edge_mask: Bool[Array, "n_edge"] | None = None,
    ) -> IrrepsArrayShape["n_node node_irreps_out"]:
        skip_connection: IrrepsArrayShape["n_node feature*hidden_irreps"] | None = None
        if self._skip_connection is not None:
            skip_connection = self._skip_connection(node_types, node_features)

        node_features = self._interaction_block(
            node_features,
            edge_features,
            radial_embedding,
            senders,
            receivers,
            edge_mask=edge_mask,
            node_types=node_types,
        )

        node_features = self._product_basis(node_features=node_features, node_types=node_types)

        if self.soft_normalisation is not None:
            node_features = e3j.norm_activation(
                node_features, [self._phi] * len(node_features.irreps)
            )

        if skip_connection is not None:
            node_features = node_features + skip_connection

        return node_features

    def _phi(self, n):
        n = n / self.soft_normalisation
        return 1.0 / (1.0 + n * e3j.sus(n))


class Mace(linen.Module):
    irreps_out: IntoIrreps
    out_field: str
    hidden_irreps: IntoIrreps  # 256x0e or 128x0e + 128x1o

    correlation_order: int = 3  # Correlation order at each layer (~ node_features^correlation)
    num_interactions: int = 2  # Number of interactions (layers)
    y0_values: list[float] | None = None
    soft_normalisation: bool | None = None
    # Number of features per node, default gcd of hidden_irreps multiplicities
    num_features: int | None = None
    num_types: int = 1
    max_ell: int = 3  # Max spherical harmonic degree
    epsilon: float | None = None

    # Normalistaion
    avg_num_neighbours: float | dict[int, float] = 1.0
    off_diagonal: bool = False

    symmetric_tensor_product_basis: bool = True
    readout_mlp_irreps: IntoIrreps = "16x0e"
    interaction_irreps: Literal["o3_restricted", "o3_full"] | IntoIrreps = "o3_restricted"

    # Radial
    radial_activation: Callable = jax.nn.silu  # activation function

    skip_connection_first_layer: bool = False

    def setup(self):
        # pylint: disable=attribute-defined-outside-init
        hidden_irreps, irreps_out, non_scalar_irreps_out = self._init_irreps(
            self.hidden_irreps, self.irreps_out
        )
        self._y0 = self._init_y0(self.y0_values, irreps_out, num_types=self.num_types)

        readout_mlp_irreps = (
            e3j.Irreps(self.readout_mlp_irreps) + non_scalar_irreps_out
        ).simplify()

        if self.num_features is None:
            num_features = functools.reduce(math.gcd, (mul for mul, _ in hidden_irreps))
            hidden_irreps = e3j.Irreps([(mul // num_features, ir) for mul, ir in hidden_irreps])
        else:
            num_features = self.num_features

        if self.interaction_irreps == "o3_restricted":
            self._interaction_irreps = e3j.Irreps.spherical_harmonics(self.max_ell)
        elif self.interaction_irreps == "o3_full":
            self._interaction_irreps = e3j.Irreps(e3j.Irrep.iterator(self.max_ell))
        else:
            self._interaction_irreps = e3j.Irreps(self.interaction_irreps)

        # Build the layers we will use
        mace_layers = []
        readouts = []
        for i in range(self.num_interactions):
            is_not_first = i != 0
            is_not_last = i != self.num_interactions - 1

            # Mace
            mace_layer = MaceLayer(
                irreps_out=irreps_out,
                num_types=self.num_types,
                # Interaction
                num_features=num_features,
                interaction_irreps=self._interaction_irreps,
                # Radial
                radial_activation=self.radial_activation,
                # Normalisation
                avg_num_neighbours=self.avg_num_neighbours,
                epsilon=self.epsilon,
                # Product basis
                hidden_irreps=hidden_irreps,
                correlation_order=self.correlation_order,
                symmetric_tensor_product_basis=self.symmetric_tensor_product_basis,
                off_diagonal=self.off_diagonal,
                # Residual
                soft_normalisation=self.soft_normalisation,
                skip_connection=is_not_first or self.skip_connection_first_layer,
            )

            # Readout
            if is_not_last:
                readout = e3j.flax.Linear(irreps_out, force_irreps_out=True)
            else:
                # Nonlinear readout on last layer
                readout = NonLinearReadoutBlock(
                    readout_mlp_irreps,
                    irreps_out,
                    activation=self.radial_activation,
                    gate=self.radial_activation,
                )

            mace_layers.append(mace_layer)
            readouts.append(readout)

        self._layers = mace_layers
        self._readouts = readouts

    @jt.jaxtyped(typechecker=beartype.beartype)
    @gcnn.shape_check
    @_base.shape_check
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        # Embeddings
        node_feats: IrrepsArrayShape["n_node feature*irreps"] = graph.nodes[keys.FEATURES]
        node_types = graph.nodes[keys.SPECIES][:, 0]

        # Interactions
        outputs: "list[IrrepsArrayShape['n_node output_irreps']]" = []
        # Deal with the y0 values of the expansion
        if self.y0_values is not None:
            outputs.append(self._y0[node_types])

        # Now expand up to the maximum correlation order
        for layer, readout in zip(self._layers, self._readouts):
            node_feats = layer(
                node_feats,
                # Edge features are not mutated, so just take directly from graph
                graph.edges[keys.ATTRIBUTES],
                node_types,
                graph.edges[keys.RADIAL_EMBEDDINGS],
                graph.senders,
                graph.receivers,
                edge_mask=graph.edges.get(keys.MASK),
            )
            node_outputs: IrrepsArrayShape["n_node output_irreps"] = readout(node_feats)

            outputs += [node_outputs]

        # Calculate the final output value by summing the values from each correlation order
        output = e3j.sum(e3j.stack(outputs, axis=1), axis=1)
        return (
            experimental.update_graph(graph)
            .update("nodes", {keys.FEATURES: node_feats, self.out_field: output})
            .get()
        )

    @staticmethod
    def _init_irreps(
        hidden_irreps: IntoIrreps, irreps_out: IntoIrreps
    ) -> tuple[e3j.Irreps, e3j.Irreps, e3j.Irreps]:
        hidden_irreps = e3j.Irreps(hidden_irreps)
        irreps_out = e3j.Irreps(irreps_out)
        # The nonlinear readout's hidden representation needs a channel matching every
        # non-scalar irrep in irreps_out, or NonLinearReadoutBlock's final Linear has no
        # equivariant path to it and those output components are silently zero.
        non_scalar_irreps_out = irreps_out.filter(drop=["0e", "0o"])

        # Sanity check
        missing = [
            ir for _, ir in non_scalar_irreps_out if ir not in {ir for _, ir in hidden_irreps}
        ]
        if missing:
            raise ValueError(
                f"irreps_out requires {missing} but hidden_irreps={hidden_irreps} has no matching "
                "channel — those output components would be identically zero."
            )

        return hidden_irreps, irreps_out, non_scalar_irreps_out

    @staticmethod
    def _init_y0(y0_values, irreps_out: e3j.Irreps, num_types: int):
        """
        Safely maps scalar 0-body baselines (y0) to arbitrary target irreps_out.
        """
        if y0_values is None:
            return None

        # 1. Extract raw array data cleanly
        if isinstance(y0_values, e3j.IrrepsArray):
            y0 = y0_values.array
        else:
            y0 = jnp.asarray(y0_values)

        # 2. Promote 1D [num_types] -> 2D [num_types, 1] for scalar features
        if y0.ndim == 1:
            y0 = y0[:, None]

        # 3. Validate leading dimension against num_types
        if y0.shape[0] != num_types:
            raise ValueError(
                f"y0_values leading dimension ({y0.shape[0]}) does not match num_types "
                f"({num_types})"
            )

        # 4. Embed into irreps_out (pad non-scalars with zero)
        num_scalars = sum(mul for mul, ir in irreps_out if ir == e3j.Irrep("0e"))
        if y0.shape[1] != num_scalars:
            raise ValueError(
                f"y0_values feature dimension ({y0.shape[1]}) does not match "
                f"number of 0e components in irreps_out ({num_scalars})"
            )

        full_array = jnp.zeros((num_types, irreps_out.dim), dtype=y0.dtype)

        # Place y0 into the 0e feature slots
        scalar_offset = 0
        array_offset = 0
        for mul, ir in irreps_out:
            dim = mul * ir.dim
            if ir == e3j.Irrep("0e"):
                full_array = full_array.at[:, array_offset : array_offset + dim].set(
                    y0[:, scalar_offset : scalar_offset + dim]
                )
                scalar_offset += dim
            array_offset += dim

        return e3j.IrrepsArray(irreps_out, full_array)
