from collections.abc import Callable, Iterable
import functools
import math
from typing import Optional, Union

import beartype
import e3nn_jax as e3j
from flax import linen
import jax
import jax.numpy as jnp
import jaxtyping as jt
import jraph

import tensorial
from tensorial import gcnn, nn_utils, typing

from . import _base, _experimental, _message_passing, keys

A025582 = [0, 1, 3, 7, 12, 20, 30, 44, 65, 80, 96, 122, 147, 181, 203, 251, 289]


class SymmetricContraction(linen.Module):
    """
    Symmetric tensor contraction up to a given correlation order.

    Based on implementation from:

        https://github.com/ACEsuit/mace-jax/blob/main/mace_jax/modules/symmetric_contraction.py
    """

    correlation_order: int
    keep_irrep_out: Union[str, Iterable[tensorial.typing.IrrepLike]]

    num_types: int = 1
    gradient_normalisation: Optional[Union[str, float]] = None
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
        inputs: typing.IrrepsArrayShape["n_node features irreps"],
        input_type: typing.IndexArray["n_node"],
    ) -> e3j.IrrepsArray:
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
        self,
        inputs: typing.IrrepsArrayShape["num_features irreps_in"],
        input_type: typing.IndexArray[""],
    ) -> typing.IrrepsArrayShape["num_features irreps_out"]:
        """
        This operation is parallel on the feature dimension (but each feature has its own
        parameters)
        Efficient implementation of:


            vmap(lambda w, x: FunctionalLinear(irreps_out)(
                w, concatenate([x, tensor_product(x, x), tensor_product(x, x, x), ...])))(w, x)


        up to x power ``self.correlation_order``

        :param inputs: the contraction inputs
        :param input_type: the contraction index
        :return: the contraction outputs
        """
        outputs: dict[e3j.Irrep, jt.Array] = dict()
        for order in range(self.correlation_order, 0, -1):  # correlation_order, ..., 1
            if self.off_diagonal:
                inp = jnp.roll(inputs.array, A025582[order - 1])
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
                basis_fn: jt.Float[jt.Array, "irreps_in^order multiplicity irreps_out"] = (
                    basis_fn.astype(inp.dtype)
                )

                weights: jt.Float[jt.Array, "multiplicity num_features"] = self.param(
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
                    )  # [num_features, (irreps_x.dim)^(oder-1), ir_out.dim]
                else:
                    outputs[ir_out] += jnp.einsum(
                        "...ki,kc->c...i", basis_fn, weights
                    )  # [num_features, (irreps_x.dim)^order, ir_out.dim]

            # ((w3 x + w2) x + w1) x
            #  \----------------/
            #         out (in the normal case)

            for ir_out, val in outputs.items():
                if isinstance(val, tuple):
                    outputs[ir_out] = val[1]
                    continue  # already done (special case optimisation above)

                value: jt.Float[jt.Array, "num_features irreps_in^(oder-1) irreps_out"] = (
                    jnp.einsum("c...ji,cj->c...i", outputs[ir_out], inp)
                )
                outputs[ir_out] = value

            # ((w3 x + w2) x + w1) x
            #  \-------------------/
            #           out

        irreps_out = e3j.Irreps(sorted(outputs.keys()))
        output: typing.IrrepsArrayShape["num_features irreps_out"] = e3j.from_chunks(
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
        node_features: typing.IrrepsArrayShape["n_nodes featuresXirreps"],
        node_types: typing.IndexArray["n_node"],
    ) -> e3j.IrrepsArray:
        node_features = node_features.mul_to_axis().remove_zero_chunks()
        node_features = self.symmetric_contractions(node_features, node_types)
        node_features = node_features.axis_to_mul()
        return e3j.flax.Linear(self._target_irreps)(node_features)


class InteractionBlock(linen.Module):
    irreps_out: typing.IntoIrreps
    avg_num_neighbours: Union[float, dict[int, float]] = 1.0
    radial_activation: Union[str, nn_utils.ActivationFunction] = "swish"

    def setup(self):
        # pylint: disable=attribute-defined-outside-init
        self._message_passing = _message_passing.MessagePassingConvolution(
            self.irreps_out, self.avg_num_neighbours, radial_activation=self.radial_activation
        )
        self._linear_down = e3j.flax.Linear(self.irreps_out, name="linear_down")

    @linen.compact
    @jt.jaxtyped(typechecker=beartype.beartype)
    def __call__(
        self,
        node_features: typing.IrrepsArrayShape["n_nodes node_irreps"],
        edge_features: typing.IrrepsArrayShape["n_edges edge_irreps"],
        radial_embedding: jt.Float[jnp.ndarray, "n_edges radial_embeddings"],
        senders: jt.Int[typing.ArrayType, "n_edges"],
        receivers: jt.Int[typing.ArrayType, "n_edges"],
        *,
        edge_mask: Optional[jt.Bool[typing.ArrayType, "n_edges"]] = None,
    ) -> typing.IrrepsArrayShape["n_nodes target_irreps"]:
        node_features = e3j.flax.Linear(node_features.irreps, name="linear_up")(node_features)

        node_features = self._message_passing(
            node_features, edge_features, radial_embedding, senders, receivers, edge_mask=edge_mask
        )

        node_features = self._linear_down(node_features)
        assert node_features.ndim == 2

        return node_features


class NonLinearReadoutBlock(linen.Module):
    hidden_irreps: typing.IntoIrreps
    output_irreps: typing.IntoIrreps
    activation: Optional[Callable] = None
    gate: Optional[Callable] = None

    def setup(self) -> None:
        # pylint: disable=attribute-defined-outside-init
        hidden_irreps = e3j.Irreps(self.hidden_irreps)
        output_irreps = e3j.Irreps(self.output_irreps)

        # Get multiplicity of (l > 0) irreps
        num_vectors = hidden_irreps.filter(drop=["0e", "0o"]).num_irreps
        self._linear = e3j.flax.Linear((hidden_irreps + e3j.Irreps(f"{num_vectors}x0e")).simplify())
        self._linear_out = e3j.flax.Linear(output_irreps, force_irreps_out=True)

    def __call__(
        self, inputs: typing.IrrepsArrayShape["n_node irreps"]
    ) -> typing.IrrepsArrayShape["n_nodes output_irreps"]:
        inputs = self._linear(inputs)
        inputs = e3j.gate(inputs, even_act=self.activation, even_gate_act=self.gate)
        return self._linear_out(inputs)


class MaceLayer(linen.Module):
    """
    A MACE layer composed of:
        * Interaction block
        * Normalisation
        * Product basis
        * (optional) self connection
    """

    irreps_out: typing.IntoIrreps
    num_types: int

    # Interaction
    num_features: int
    interaction_irreps: typing.IntoIrreps
    #   Radial
    radial_activation: Callable

    # Normalisation
    epsilon: Optional[float]
    avg_num_neighbours: float

    # Product basis
    hidden_irreps: typing.IntoIrreps
    correlation_order: int
    symmetric_tensor_product_basis: bool
    off_diagonal: bool

    soft_normalisation: Optional[float]
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
            irreps_out=num_features * interaction_irreps,
            avg_num_neighbours=self.avg_num_neighbours,
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
        node_features: typing.IrrepsArrayShape["n_nodes node_irreps"],
        edge_features: typing.IrrepsArrayShape["n_edges edge_irreps"],
        node_species: jt.Int[jt.Array, "n_nodes"],  # int between 0 and num_species - 1
        radial_embedding: jt.Float[jt.Array, "n_edges radial_embedding"],
        senders: jt.Int[typing.ArrayType, "n_edges"],
        receivers: jt.Int[typing.ArrayType, "n_edges"],
        *,
        edge_mask: Optional[jt.Bool[typing.ArrayType, "n_edges"]] = None,
    ) -> typing.IrrepsArrayShape["n_nodes node_irreps_out"]:
        skip_connection: Optional[typing.IrrepsArrayShape["n_nodes feature*hidden_irreps"]] = None
        if self._skip_connection is not None:
            skip_connection = self._skip_connection(node_species, node_features)

        node_features = self._interaction_block(
            node_features=node_features,
            edge_features=edge_features,
            radial_embedding=radial_embedding,
            receivers=receivers,
            senders=senders,
            edge_mask=edge_mask,
        )

        if self.epsilon is not None:
            node_features *= self.epsilon
        else:
            node_features /= jnp.sqrt(self.avg_num_neighbours)

        node_features = self._product_basis(node_features=node_features, node_types=node_species)

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
    irreps_out: typing.IntoIrreps
    out_field: str
    hidden_irreps: typing.IntoIrreps  # 256x0e or 128x0e + 128x1o

    correlation_order: int = 3  # Correlation order at each layer (~ node_features^correlation)
    num_interactions: int = 2  # Number of interactions (layers)
    y0_values: Optional[list[float]] = None
    avg_num_neighbours: Union[float, dict[int, float]] = 1.0
    soft_normalisation: Optional[bool] = None
    # Number of features per node, default gcd of hidden_irreps multiplicities
    num_features: Optional[int] = None
    num_types: int = 1
    max_ell: int = 3  # Max spherical harmonic degree
    epsilon: Optional[float] = None
    off_diagonal: bool = False

    symmetric_tensor_product_basis: bool = True
    readout_mlp_irreps: typing.IntoIrreps = "16x0e"
    interaction_irreps: Union[str, typing.IntoIrreps] = "o3_restricted"  # or o3_full

    # Radial
    radial_activation: Callable = jax.nn.silu  # activation function

    skip_connection_first_layer: bool = False

    def setup(self):
        # pylint: disable=attribute-defined-outside-init
        hidden_irreps = e3j.Irreps(self.hidden_irreps)
        irreps_out = e3j.Irreps(self.irreps_out)
        if self.y0_values:
            if len(self.y0_values) != self.num_types:
                raise ValueError(f"len(y0) != num_types: {len(self.y0_values)} != {self.num_types}")

            y0 = self.y0_values
            if isinstance(self.y0_values, e3j.IrrepsArray):
                y0 = self.y0_values.array
            elif isinstance(self.y0_values, (list, tuple)):
                y0 = jnp.array(self.y0_values)

            self._y0 = e3j.IrrepsArray(irreps_out, jnp.atleast_2d(y0).T)

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
                epsilon=self.epsilon,
                avg_num_neighbours=self.avg_num_neighbours,
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
                    e3j.Irreps(self.readout_mlp_irreps),
                    irreps_out,
                    activation=self.radial_activation,
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
        node_feats: typing.IrrepsArrayShape["n_nodes feature*irreps"] = graph.nodes[keys.FEATURES]
        node_species = graph.nodes[keys.SPECIES]

        # Interactions
        outputs: list[typing.IrrepsArrayShape["n_nodes output_irreps"]] = []
        # Deal with the y0 values of the expansion
        if self.y0_values:
            outputs.append(self._y0[node_species])

        # Now expand up to the maximum correlation order
        for layer, readout in zip(self._layers, self._readouts):
            node_feats = layer(
                node_feats,
                # Edge features are not mutated, so just take directly from graph
                graph.edges[keys.ATTRIBUTES],
                node_species,
                graph.edges[keys.RADIAL_EMBEDDINGS],
                graph.senders,
                graph.receivers,
                edge_mask=graph.edges.get(keys.MASK),
            )
            node_outputs: typing.IrrepsArrayShape["n_nodes output_irreps"] = readout(node_feats)

            outputs += [node_outputs]

        return (
            _experimental.update_graph(graph)
            .update(
                "nodes",
                {
                    keys.FEATURES: node_feats,
                    self.out_field: e3j.sum(e3j.stack(outputs, axis=1), axis=1),
                },
            )
            .get()
        )
        # updates = utils.UpdateGraphDicts(graph)
        # updates.nodes[keys.FEATURES] = node_feats
        # updates.nodes[self.out_field] = e3j.sum(e3j.stack(outputs, axis=1), axis=1)

        # return updates.get()
