# -*- coding: utf-8 -*-
import functools
import math
from typing import Callable, Optional, Set, Tuple, Union

import e3nn_jax as e3j
from flax import linen
import jax
import jax.numpy as jnp
import jraph

from tensorial import nn_utils

from . import _message_passing, keys

A025582 = [0, 1, 3, 7, 12, 20, 30, 44, 65, 80, 96, 122, 147, 181, 203, 251, 289]


class SymmetricContraction(linen.Module):
    correlation_order: int
    keep_irrep_out: Set[e3j.Irrep]
    num_species: int
    gradient_normalization: Union[str, float] = None
    symmetric_tensor_product_basis: bool = True
    off_diagonal: bool = False
    param_dtype = jnp.float32

    def setup(self):
        # pylint: disable=attribute-defined-outside-init
        # Gradient normalisation
        gradient_normalization = self.gradient_normalization
        if gradient_normalization is None:
            gradient_normalization = e3j.config("gradient_normalization")
        if isinstance(gradient_normalization, str):
            gradient_normalization = {"element": 0.0, "path": 1.0}[gradient_normalization]
        self._gradient_normalization = gradient_normalization

        # Output irreps to keep
        keep_irrep_out = (
            e3j.Irreps(self.keep_irrep_out)
            if isinstance(self.keep_irrep_out, str)
            else self.keep_irrep_out
        )
        assert all(mul == 1 for mul, _ in keep_irrep_out)
        self._keep_irrep_out = {e3j.Irrep(ir) for ir in keep_irrep_out}

    def __call__(self, inputs: e3j.IrrepsArray, index: jnp.ndarray) -> e3j.IrrepsArray:
        # Treat batch indices using vmap
        shape = jnp.broadcast_shapes(inputs.shape[:-2], index.shape)
        inputs = inputs.broadcast_to(shape + inputs.shape[-2:])
        index = jnp.broadcast_to(index, shape)

        fn_mapped = self.contract
        for _ in range(inputs.ndim - 2):
            fn_mapped = jax.vmap(fn_mapped)

        return fn_mapped(inputs, index)

    def contract(self, inputs: e3j.IrrepsArray, index: jnp.ndarray):
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
        assert inputs.ndim == 2  # [num_features, irreps_x.dim]
        assert index.ndim == 0  # int

        outputs = dict()
        for order in range(self.correlation_order, 0, -1):  # correlation, ..., 1
            if self.off_diagonal:
                x_ = jnp.roll(inputs.array, A025582[order - 1])
            else:
                x_ = inputs.array

            if self.symmetric_tensor_product_basis:
                basis = e3j.reduced_symmetric_tensor_product_basis(
                    inputs.irreps, order, keep_ir=self._keep_irrep_out
                )
            else:
                basis = e3j.reduced_tensor_product_basis(
                    [inputs.irreps] * order, keep_ir=self._keep_irrep_out
                )
            # U = U / order  # normalization TODO(mario): put back after testing
            # NOTE(mario): The normalization constants (/order and /mul**0.5)
            # has been numerically checked to be correct.

            # TODO(mario) implement norm_p

            # ((w3 x + w2) x + w1) x
            #  \-----------/
            #       out

            for (mul, ir_out), basis_fn in zip(basis.irreps, basis.chunks):
                basis_fn = basis_fn.astype(x_.dtype)
                # basis: ndarray [(irreps_x.dim)^order, multiplicity, ir_out.dim]

                weights = self.param(
                    f"w{order}_{ir_out}",
                    linen.initializers.normal(
                        stddev=(mul**-0.5) ** (1.0 - self._gradient_normalization)
                    ),
                    (self.num_species, mul, inputs.shape[0]),
                    self.param_dtype,
                )[
                    index
                ]  # [multiplicity, num_features]

                # normalize weights
                weights = weights * (mul**-0.5) ** self._gradient_normalization

                if ir_out not in outputs:
                    outputs[ir_out] = (
                        "special",
                        jnp.einsum("...jki,kc,cj->c...i", basis_fn, weights, x_),
                    )  # [num_features, (irreps_x.dim)^(oder-1), ir_out.dim]
                else:
                    outputs[ir_out] += jnp.einsum(
                        "...ki,kc->c...i", basis_fn, weights
                    )  # [num_features, (irreps_x.dim)^order, ir_out.dim]

            # ((w3 x + w2) x + w1) x
            #  \----------------/
            #         out (in the normal case)

            for ir_out in outputs:
                if isinstance(outputs[ir_out], tuple):
                    outputs[ir_out] = outputs[ir_out][1]
                    continue  # already done (special case optimization above)

                outputs[ir_out] = jnp.einsum(
                    "c...ji,cj->c...i", outputs[ir_out], x_
                )  # [num_features, (irreps_x.dim)^(oder-1), ir_out.dim]

            # ((w3 x + w2) x + w1) x
            #  \-------------------/
            #           out

        # out[irrep_out] : [num_features, ir_out.dim]
        irreps_out = e3j.Irreps(sorted(outputs.keys()))
        return e3j.from_chunks(
            irreps_out,
            [outputs[ir][:, None, :] for (_, ir) in irreps_out],
            (inputs.shape[0],),
        )


class EquivariantProductBasisBlock(linen.Module):
    irreps_out: e3j.Irreps
    correlation_order: int
    num_species: int
    symmetric_tensor_product_basis: bool = True
    off_diagonal: bool = False

    def setup(self):
        # pylint: disable=attribute-defined-outside-init
        self._target_irreps = e3j.Irreps(self.irreps_out)
        self.symmetric_contractions = SymmetricContraction(
            keep_irrep_out={ir for _, ir in e3j.Irreps(self._target_irreps)},
            correlation=self.correlation_order,
            num_species=self.num_species,
            gradient_normalization="element",  # NOTE: This is to copy mace-torch
            symmetric_tensor_product_basis=self.symmetric_tensor_product_basis,
            off_diagonal=self.off_diagonal,
        )

    def __call__(
        self,
        node_feats: e3j.IrrepsArray,  # [n_nodes, feature * irreps]
        node_specie: jnp.ndarray,  # [n_nodes, ] int
    ) -> e3j.IrrepsArray:
        node_feats = node_feats.mul_to_axis().remove_zero_chunks()
        node_feats = self.symmetric_contractions(node_feats, node_specie)
        node_feats = node_feats.axis_to_mul()
        return e3j.flax.Linear(self._target_irreps)(node_feats)


class InteractionBlock(linen.Module):
    irreps_out: e3j.Irreps
    avg_num_neighbours: float = 1.0
    radial_activation: nn_utils.ActivationFunction

    def setup(self):
        # pylint: disable=attribute-defined-outside-init
        self._linear_down = e3j.flax.Linear(self.irreps_out, name="linear_down")
        self._message_passing = _message_passing.MessagePassingConvolution(
            self.irreps_out, self.avg_num_neighbors, radial_activation=self.radial_activation
        )

    def __call__(
        self,
        node_feats: e3j.IrrepsArray,  # [n_nodes, irreps]
        edge_features: e3j.IrrepsArray,  # [n_edges, 3]
        radial_embedding: jnp.ndarray,  # [n_edges, radial_embedding_dim]
        senders: jnp.ndarray,  # [n_edges, ]
        receivers: jnp.ndarray,  # [n_edges, ]
        edge_mask: Optional[jax.Array] = None,
    ) -> Tuple[e3j.IrrepsArray, e3j.IrrepsArray]:
        assert node_feats.ndim == 2
        if not edge_features.ndim:
            raise ValueError(
                f"Expected edge attributes to have two dimensions, got {edge_features.ndim}"
            )
        if not radial_embedding.ndim == 2:
            raise ValueError(
                f"Expected radial embedding to have two dimensions, got {radial_embedding.ndim}"
            )

        node_feats = e3j.flax.Linear(node_feats.irreps, name="linear_up")(node_feats)

        node_feats = self._message_passing(
            node_feats, edge_features, radial_embedding, senders, receivers, edge_mask
        )

        node_feats = self._linear_down(node_feats)
        assert node_feats.ndim == 2

        return node_feats  # [n_nodes, target_irreps]


class NonLinearReadoutBlock(linen.Module):
    hidden_irreps: e3j.Irreps
    output_irreps: e3j.Irreps
    activation: Optional[Callable] = None
    gate: Optional[Callable] = None

    def setup(self) -> None:
        # pylint: disable=attribute-defined-outside-init
        # Get multiplicity of (l > 0) irreps
        num_vectors = self.hidden_irreps.filter(drop=["0e", "0o"]).num_irreps
        self._linear = e3j.flax.Linear(
            (self.hidden_irreps + e3j.Irreps(f"{num_vectors}x0e")).simplify()
        )

    def __call__(self, inputs: e3j.IrrepsArray) -> e3j.IrrepsArray:
        # inputs = [n_nodes, irreps]
        inputs = self._linear(inputs)
        inputs = e3j.gate(inputs, even_act=self.activation, even_gate_act=self.gate)
        return e3j.haiku.Linear(self.output_irreps)(inputs)  # [n_nodes, output_irreps]


class MACELayer(linen.Module):
    first: bool
    num_features: int
    interaction_irreps: e3j.Irreps
    hidden_irreps: e3j.Irreps
    correlation_order: int
    avg_num_neighbors: float
    # Radial
    radial_activation: Callable

    num_species: int

    # Interaction block
    epsilon: Optional[float]

    # EquivariantProductBasisBlock:
    symmetric_tensor_product_basis: bool
    off_diagonal: bool
    soft_normalization: Optional[float]

    # ReadoutBlock:
    output_irreps: e3j.Irreps
    readout_mlp_irreps: e3j.Irreps
    skip_connection_first_layer: bool = False

    def setup(self):
        # pylint: disable=attribute-defined-outside-init
        self._interaction_block = InteractionBlock(
            irreps_out=self.num_features * self.interaction_irreps,
            avg_num_neighbors=self.avg_num_neighbors,
            radial_activation=self.radial_activation,
        )
        self._product_basis = EquivariantProductBasisBlock(
            irreps_out=self.num_features * self.hidden_irreps,
            correlation_order=self.correlation_order,
            num_species=self.num_species,
            symmetric_tensor_product_basis=self.symmetric_tensor_product_basis,
            off_diagonal=self.off_diagonal,
        )
        self._self_connection = e3j.flax.Linear(
            self.num_features * self.hidden_irreps,
            num_indexed_weights=self.num_species,
            name="self_connection",
        )

    def __call__(
        self,
        node_feats: e3j.IrrepsArray,  # [n_nodes, irreps]
        edge_features: e3j.IrrepsArray,  # [n_edges, 3]
        node_specie: jnp.ndarray,  # [n_nodes] int between 0 and num_species-1
        radial_embedding: jnp.ndarray,  # [n_edges, radial_embedding_dim]
        senders: jax.Array,  # [n_edges]
        receivers: jax.Array,  # [n_edges]
        node_mask: jax.Array = None,  # [n_node]
    ):
        sc = None
        if not self.first or self.skip_connection_first_layer:
            sc = self._self_connection(
                node_specie, node_feats
            )  # [n_nodes, feature * hidden_irreps]

        node_feats = self._interaction_block(
            node_feats=node_feats,
            edge_features=edge_features,
            radial_embedding=radial_embedding,
            receivers=receivers,
            senders=senders,
            node_mask=node_mask,
        )

        if self.epsilon is not None:
            node_feats *= self.epsilon
        else:
            node_feats /= jnp.sqrt(self.avg_num_neighbors)

        if self.first:
            # Selector TensorProduct
            node_feats = e3j.flax.Linear(
                self.num_features * self.interaction_irreps,
                num_indexed_weights=self.num_species,
                name="skip_tp_first",
            )(node_specie, node_feats)

        node_feats = self._product_basis(node_feats=node_feats, node_specie=node_specie)

        if self.soft_normalization is not None:

            def phi(n):
                n = n / self.soft_normalization
                return 1.0 / (1.0 + n * e3j.sus(n))

            node_feats = e3j.norm_activation(node_feats, [phi] * len(node_feats.irreps))

        if sc is not None:
            node_feats = node_feats + sc  # [n_nodes, feature * hidden_irreps]

        node_outputs = e3j.flax.Linear(self.output_irreps)(node_feats)  # [n_nodes, output_irreps]
        return node_outputs, node_feats


class MACE(linen.Module):
    correlation_order: int = (
        3  # Correlation order at each layer (~ node_features^correlation), default 3
    )
    num_interactions: int  # Number of interactions (layers), default 2
    avg_num_neighbors: float
    num_features: int = (
        None,
    )  # Number of features per node, default gcd of hidden_irreps multiplicities
    hidden_irreps: e3j.Irreps  # 256x0e or 128x0e + 128x1o
    num_species: int = 1
    max_ell: int = 3  # Max spherical harmonic degree, default 3
    epsilon: Optional[float] = None
    off_diagonal: bool = False
    activation: Callable = jax.nn.silu  # activation function

    def __init__(
        self,
        *,
        readout_mlp_irreps: e3j.Irreps,  # Hidden irreps of the MLP in last readout, default 16x0e
        # Number of zero derivatives at small and large distances, default 4 and 2
        # If both are None, it uses a smooth C^inf envelope function
        soft_normalization: Optional[float] = None,
        symmetric_tensor_product_basis: bool = True,
        interaction_irreps: Union[str, e3j.Irreps] = "o3_restricted",  # or o3_full
        skip_connection_first_layer: bool = False,
    ):
        super().__init__()

        readout_mlp_irreps = e3j.Irreps(readout_mlp_irreps)

        if self.num_features is None:
            self._num_features = functools.reduce(math.gcd, (mul for mul, _ in self.hidden_irreps))
            self._hidden_irreps = e3j.Irreps(
                [(mul // self.num_features, ir) for mul, ir in self.hidden_irreps]
            )
        else:
            self._num_features = self.num_features
            self._hidden_irreps = self.hidden_irreps

        if interaction_irreps == "o3_restricted":
            self.interaction_irreps = e3j.Irreps.spherical_harmonics(self.max_ell)
        elif interaction_irreps == "o3_full":
            self.interaction_irreps = e3j.Irreps(e3j.Irrep.iterator(self.max_ell))
        else:
            self.interaction_irreps = e3j.Irreps(interaction_irreps)

        self.readout_mlp_irreps = readout_mlp_irreps
        self.symmetric_tensor_product_basis = symmetric_tensor_product_basis
        self.soft_normalization = soft_normalization
        self.skip_connection_first_layer = skip_connection_first_layer

    def __call__(self, graph: jraph.GraphsTuple) -> e3j.IrrepsArray:
        # Embeddings
        node_feats = graph.nodes[keys.FEATURES]  # [n_nodes, feature * irreps]
        radial_embedding = graph.edges[keys.RADIAL_EMBEDDINGS]

        # Interactions
        outputs = []
        for i in range(self.num_interactions):
            first = i == 0
            last = i == self.num_interactions - 1

            node_outputs, node_feats = MACELayer(
                first=first,
                last=last,
                num_features=self._num_features,
                interaction_irreps=self.interaction_irreps,
                hidden_irreps=self._hidden_irreps,
                avg_num_neighbors=self.avg_num_neighbors,
                activation=self.activation,
                num_species=self.num_species,
                epsilon=self.epsilon,
                correlation=self.correlation_order,
                output_irreps=self.output_irreps,
                readout_mlp_irreps=self.readout_mlp_irreps,
                symmetric_tensor_product_basis=self.symmetric_tensor_product_basis,
                off_diagonal=self.off_diagonal,
                soft_normalization=self.soft_normalization,
                skip_connection_first_layer=self.skip_connection_first_layer,
                name=f"layer_{i}",
            )(
                node_feats,
                graph.edges[keys.FEATURES],
                graph.nodes[keys.SPECIES],
                radial_embedding,
                graph.senders,
                graph.receivers,
                node_mask=graph.nodes.get(keys.MASK),
            )
            outputs += [node_outputs]  # list of [n_nodes, output_irreps]

        return e3j.stack(outputs, axis=1)  # [n_nodes, num_interactions, output_irreps]
