# -*- coding: utf-8 -*-
import functools
import math
from typing import Callable, Dict, Optional, Set, Union

import beartype
import e3nn_jax as e3j
from flax import linen
import jax
import jax.numpy as jnp
import jaxtyping as jt
import jraph

import tensorial
from tensorial import nn_utils

from . import _message_passing, keys, typing, utils

A025582 = [0, 1, 3, 7, 12, 20, 30, 44, 65, 80, 96, 122, 147, 181, 203, 251, 289]


class SymmetricContraction(linen.Module):
    correlation_order: int
    keep_irrep_out: Set[e3j.Irrep]

    num_types: int = 1
    gradient_normalisation: Union[str, float] = None
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
    def __call__(
        self, inputs: jt.Float[e3j.IrrepsArray, "batch features irreps"], index: typing.IndexArray
    ) -> e3j.IrrepsArray:
        # Treat batch indices using vmap
        shape = jnp.broadcast_shapes(inputs.shape[:-2], index.shape)
        inputs = inputs.broadcast_to(shape + inputs.shape[-2:])
        index = jnp.broadcast_to(index, shape)

        contract = self._contract
        for _ in range(inputs.ndim - 2):
            contract = jax.vmap(contract)

        return contract(inputs, index)

    def _contract(
        self,
        inputs: jt.Float[e3j.IrrepsArray, "num_features in_irreps"],
        index: jt.Int[jax.Array, ""],
    ):
        """
        This operation is parallel on the feature dimension (but each feature has its own
        parameters)
        This operation is an efficient implementation of:


            vmap(lambda w, x: FunctionalLinear(irreps_out)(
                w, concatenate([x, tensor_product(x, x), tensor_product(x, x, x), ...])))(w, x)


        up to x power self.correlation

        TODO: Rewrite this
        :param inputs:
        :param index:
        :return:
        """
        outputs: Dict[e3j.Irrep, jax.Array] = dict()
        for order in range(self.correlation_order, 0, -1):  # correlation, ..., 1
            if self.off_diagonal:
                inp = jnp.roll(inputs.array, A025582[order - 1])
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

            # ((w3 x + w2) x + w1) x
            #  \-----------/
            #       out

            for (mul, ir_out), basis_fn in zip(basis.irreps, basis.chunks):
                basis_fn: jt.Float[jax.Array, "in_irreps^order multiplicity out_irreps"] = (
                    basis_fn.astype(inp.dtype)
                )

                weights: jt.Float[jax.Array, "multiplicity num_features"] = (
                    self.param(  # pylint: disable=unsubscriptable-object
                        f"w{order}_{ir_out}",
                        linen.initializers.normal(
                            stddev=(mul**-0.5) ** (1.0 - self._gradient_normalisation)
                        ),
                        (self.num_types, mul, inputs.shape[0]),
                        self.param_dtype,
                    )[index]
                )

                # normalize weights
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

                value: jt.Float[jax.Array, "num_features (in_irreps)^(oder-1) out_irreps"] = (
                    jnp.einsum("c...ji,cj->c...i", outputs[ir_out], inp)
                )
                outputs[ir_out] = value

            # ((w3 x + w2) x + w1) x
            #  \-------------------/
            #           out

        irreps_out = e3j.Irreps(sorted(outputs.keys()))
        output: jt.Float[e3j.IrrepsArray, "num_features out_irreps"] = e3j.from_chunks(
            irreps_out,
            [outputs[ir][:, None, :] for (_, ir) in irreps_out],
            (inputs.shape[0],),
        )
        return output


@jt.jaxtyped(beartype.beartype)
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
    def __call__(
        self,
        node_features: jt.Float[e3j.IrrepsArray, "n_nodes featureXirreps"],
        node_types: jt.Int[jax.Array, "n_node"],
    ) -> e3j.IrrepsArray:
        node_features = node_features.mul_to_axis().remove_zero_chunks()
        node_features = self.symmetric_contractions(node_features, node_types)
        node_features = node_features.axis_to_mul()
        return e3j.flax.Linear(self._target_irreps)(node_features)


@jt.jaxtyped(beartype.beartype)
class InteractionBlock(linen.Module):
    irreps_out: e3j.Irreps
    avg_num_neighbours: float = 1.0
    radial_activation: nn_utils.ActivationFunction = "swish"

    def setup(self):
        # pylint: disable=attribute-defined-outside-init
        self._linear_down = e3j.flax.Linear(self.irreps_out, name="linear_down")
        self._message_passing = _message_passing.MessagePassingConvolution(
            self.irreps_out, self.avg_num_neighbours, radial_activation=self.radial_activation
        )

    @linen.compact
    def __call__(
        self,
        node_features: jt.Float[e3j.IrrepsArray, "n_nodes irreps"],
        edge_features: jt.Float[e3j.IrrepsArray, "n_edges irreps"],
        radial_embedding: jt.Float[jnp.ndarray, "n_edges radial_embeddings"],
        senders: jt.Int[jax.Array, "n_edges"],
        receivers: jt.Int[jax.Array, "n_edges"],
        edge_mask: Optional[jt.Bool[jax.Array, "n_edges"]] = None,
    ) -> e3j.IrrepsArray:
        assert node_features.ndim == 2
        if not edge_features.ndim:
            raise ValueError(
                f"Expected edge attributes to have two dimensions, got {edge_features.ndim}"
            )
        if not radial_embedding.ndim == 2:
            raise ValueError(
                f"Expected radial embedding to have two dimensions, got {radial_embedding.ndim}"
            )

        node_features = e3j.flax.Linear(node_features.irreps, name="linear_up")(node_features)

        node_features = self._message_passing(
            node_features, edge_features, radial_embedding, senders, receivers, edge_mask
        )

        node_features = self._linear_down(node_features)
        assert node_features.ndim == 2

        return node_features  # [n_nodes, target_irreps]


class NonLinearReadoutBlock(linen.Module):
    hidden_irreps: tensorial.typing.IrrepsLike
    output_irreps: tensorial.typing.IrrepsLike
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
        self, inputs: jt.Float[e3j.IrrepsArray, "n_node irreps"]
    ) -> jt.Float[e3j.IrrepsArray, "n_nodes output_irreps"]:
        inputs = self._linear(inputs)
        inputs = e3j.gate(inputs, even_act=self.activation, even_gate_act=self.gate)
        return self._linear_out(inputs)


@jt.jaxtyped(beartype.beartype)
class MaceLayer(linen.Module):
    """
    A MACE layer composed of:
        * Interaction block
        * Normalisation
        * Product basis
        * (optional) self connection
    """

    irreps_out: e3j.Irreps
    num_types: int

    # Interaction
    num_features: int
    interaction_irreps: tensorial.typing.IrrepsLike
    #   radial
    radial_activation: Callable

    # Normalisation
    epsilon: Optional[float]
    avg_num_neighbours: float

    # Product basis
    hidden_irreps: tensorial.typing.IrrepsLike
    correlation_order: int
    symmetric_tensor_product_basis: bool
    off_diagonal: bool

    soft_normalisation: Optional[float]
    self_connection: bool = True

    def setup(self):
        # pylint: disable=attribute-defined-outside-init
        hidden_irreps = e3j.Irreps(self.hidden_irreps)
        interaction_irreps = e3j.Irreps(self.hidden_irreps)

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
        if self.self_connection:
            self._self_connection = e3j.flax.Linear(
                num_features * hidden_irreps,
                num_indexed_weights=self.num_types,
                name="self_connection",
                force_irreps_out=True,
            )
        else:
            self._self_connection = None

    def __call__(
        self,
        node_features: jt.Float[e3j.IrrepsArray, "n_nodes irreps"],
        edge_features: jt.Float[e3j.IrrepsArray, "n_edges edge_irreps"],
        node_species: jt.Int[jax.Array, "n_nodes"],  # int between 0 and num_species - 1
        radial_embedding: jt.Float[jax.Array, "edge radial_embedding"],
        senders: jt.Int[jax.Array, "n_edges"],
        receivers: jt.Int[jax.Array, "n_edges"],
        edge_mask: Optional[jt.Bool[jax.Array, "n_edges"]] = None,
    ) -> e3j.IrrepsArray:
        self_connection = None
        if self._self_connection is not None:
            self_connection = self._self_connection(
                node_species, node_features
            )  # [n_nodes, feature * hidden_irreps]

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

        if self_connection is not None:
            node_features = node_features + self_connection

        return node_features

    def _phi(self, n):
        n = n / self.soft_normalisation
        return 1.0 / (1.0 + n * e3j.sus(n))


@jt.jaxtyped(beartype.beartype)
class Mace(linen.Module):
    irreps_out: tensorial.typing.IrrepsLike
    out_field: str
    hidden_irreps: tensorial.typing.IrrepsLike  # 256x0e or 128x0e + 128x1o

    correlation_order: int = 3  # Correlation order at each layer (~ node_features^correlation)
    num_interactions: int = 2  # Number of interactions (layers)
    avg_num_neighbours: float = 1.0
    soft_normalisation: Optional[bool] = None
    # Number of features per node, default gcd of hidden_irreps multiplicities
    num_features: int = None
    num_types: int = 1
    max_ell: int = 3  # Max spherical harmonic degree
    epsilon: Optional[float] = None
    off_diagonal: bool = False

    symmetric_tensor_product_basis: bool = True
    readout_mlp_irreps: tensorial.typing.IrrepsLike = "16x0e"
    interaction_irreps: Union[str, tensorial.typing.IrrepsLike] = "o3_restricted"  # or o3_full

    # Radial
    radial_activation: Callable = jax.nn.silu  # activation function

    skip_connection_first_layer: bool = False

    def setup(self):
        # pylint: disable=attribute-defined-outside-init
        hidden_irreps = e3j.Irreps(self.hidden_irreps)
        irreps_out = e3j.Irreps(self.irreps_out)

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
                #   radial
                radial_activation=self.radial_activation,
                # Normalisation
                epsilon=self.epsilon,
                avg_num_neighbours=self.avg_num_neighbours,
                # Product basis
                hidden_irreps=hidden_irreps,
                correlation_order=self.correlation_order,
                symmetric_tensor_product_basis=self.symmetric_tensor_product_basis,
                off_diagonal=self.off_diagonal,
                # Radial
                soft_normalisation=self.soft_normalisation,
                self_connection=is_not_first or self.skip_connection_first_layer,
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

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        # Embeddings
        node_feats = graph.nodes[keys.FEATURES]  # [n_nodes, feature * irreps]
        node_species = graph.nodes[keys.SPECIES]

        # Interactions
        outputs = []
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
            node_outputs = readout(node_feats)

            outputs += [node_outputs]  # List[[n_nodes, output_irreps]]

        updates = utils.UpdateDict(graph._asdict())
        updates["nodes"][keys.FEATURES] = node_feats
        updates["nodes"][self.out_field] = e3j.sum(e3j.stack(outputs, axis=1), axis=1)

        return graph._replace(**updates._asdict())
