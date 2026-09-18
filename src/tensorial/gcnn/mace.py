from collections.abc import Callable
import functools
import logging
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

from . import _base, _message_passing, _product_basis, experimental, keys
from .. import nn_utils
from ..typing import IntoIrreps, IrrepsArrayShape

__all__ = "Mace", "MaceLayer", "InteractionBlock", "NonLinearReadoutBlock"

_LOGGER = logging.getLogger(__name__)


def broadcast_to_nodes(
    graph_array: IrrepsArrayShape["n_graph irreps"],
    n_node: Int[Array, "n_graph"],
    total_nodes: int,
) -> IrrepsArrayShape["n_node irreps"]:
    """Repeat a per-graph quantity out to one copy per node, in node order."""
    repeated = jnp.repeat(graph_array.array, n_node, axis=0, total_repeat_length=total_nodes)
    return e3j.IrrepsArray(graph_array.irreps, repeated)


class InteractionBlock(linen.Module):
    irreps_out: IntoIrreps

    # Normalisation
    avg_num_neighbours: float | dict[int, float] = 1.0
    epsilon: float | None = None

    radial_activation: str | nn_utils.ActivationFunction = "swish"

    def setup(self):
        # pylint: disable=attribute-defined-outside-init
        self._target_irreps = e3j.Irreps(self.irreps_out)

        self._message_passing = _message_passing.MessagePassingConvolution(
            self._target_irreps,
            avg_num_neighbours=self.avg_num_neighbours,
            epsilon=self.epsilon,
            radial_activation=self.radial_activation,
        )
        self._linear_down = e3j.flax.Linear(
            self._target_irreps, name="linear_down", force_irreps_out=True
        )

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
    global_interaction: bool = False
    global_correlation_order: int | None = None
    global_irreps_out: IntoIrreps | None = None

    soft_normalisation: float | None = None
    skip_connection: bool = True

    def setup(self):
        # pylint: disable=attribute-defined-outside-init
        interaction_irreps = e3j.Irreps(self.interaction_irreps)
        hidden_irreps = e3j.Irreps(self.hidden_irreps)

        if self.num_features is None:
            num_features = functools.reduce(math.gcd, (mul for mul, _ in hidden_irreps))
            hidden_irreps = e3j.Irreps([(mul // num_features, ir) for mul, ir in hidden_irreps])
        else:
            num_features = self.num_features

        self._target_irreps: e3j.Irreps = num_features * hidden_irreps

        self._interaction_block = InteractionBlock(
            num_features * interaction_irreps,
            avg_num_neighbours=self.avg_num_neighbours,
            epsilon=self.epsilon,
            radial_activation=self.radial_activation,
        )

        self._product_basis = _product_basis.EquivariantProductBasisBlock(
            self._target_irreps,
            correlation_order=self.correlation_order,
            num_types=self.num_types,
            symmetric_tensor_product_basis=self.symmetric_tensor_product_basis,
            off_diagonal=self.off_diagonal,
        )

        if self.global_interaction:
            global_correlation_order = (
                self.global_correlation_order
                if self.global_correlation_order is not None
                else self.correlation_order
            )
            global_irreps = (
                e3j.Irreps(self.global_irreps_out)
                if self.global_irreps_out is not None
                else self._target_irreps
            )
            global_product_basis, node_global = self._init_global_interaction(
                global_irreps,
                global_correlation_order,
                self.symmetric_tensor_product_basis,
                self._target_irreps,
            )

            self._global_product_basis = global_product_basis
            self._node_global = node_global
        else:
            self._global_product_basis = None
            self._node_global = None

        if self.skip_connection:
            self._skip_connection = e3j.flax.Linear(
                self._target_irreps,
                num_indexed_weights=self.num_types,
                name="skip_connection",
                force_irreps_out=True,
            )
        else:
            self._skip_connection = None

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __call__(
        self,
        # 1. Node data
        node_features: IrrepsArrayShape["n_node node_irreps"],
        node_types: Int[Array, "n_node"],
        # 2. Edge data & geometry
        edge_features: IrrepsArrayShape["n_edge edge_irreps"],
        radial_embedding: Float[Array, "n_edge radial_embedding"],
        # 3. Graph topology
        senders: Int[Array, "n_edge"],
        receivers: Int[Array, "n_edge"],
        n_node: Int[Array, "n_graph"],
        *,
        # 4. Globals & Optionals
        global_features: IrrepsArrayShape["n_graph global_irreps"] | None = None,
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

        node_features = self._product_basis(node_features, input_type=node_types)
        if self.global_interaction:
            if global_features is None:
                raise ValueError(
                    "MaceLayer was configured with global_interaction=True but "
                    "received global_features=None"
                )
            global_features = self._global_product_basis(global_features)
            global_features_nodes = broadcast_to_nodes(
                global_features, n_node, total_nodes=node_features.shape[0]
            )

            # Cross correlate node and global features
            cross = e3j.tensor_product(
                node_features, global_features_nodes, filter_ir_out=self._target_irreps
            )
            cross = self._node_global(cross)

            node_features = node_features + cross

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

    @staticmethod
    def _init_global_interaction(
        global_irreps: e3j.Irreps,
        global_correlation_order: int,
        symmetric_tensor_product_basis: bool,
        target_irreps: e3j.Irreps,
    ) -> tuple[_product_basis.EquivariantProductBasisBlock, e3j.flax.Linear]:

        cross_irreps: e3j.Irreps = e3j.tensor_product(
            target_irreps, global_irreps, filter_ir_out=target_irreps
        )
        if cross_irreps.dim == 0:
            raise ValueError(
                f"global_irreps_out={global_irreps} has no tensor-product path to "
                f"target_irreps={target_irreps}; the global cross term would be identically zero."
            )

        global_product_basis = _product_basis.EquivariantProductBasisBlock(
            global_irreps,
            correlation_order=global_correlation_order,
            num_types=1,  # one global value per graph, not per species
            symmetric_tensor_product_basis=symmetric_tensor_product_basis,
        )
        node_global = e3j.flax.Linear(target_irreps, name="linear_cross", force_irreps_out=True)

        return global_product_basis, node_global


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

    global_interaction: bool = False
    global_correlation_order: int | None = None
    global_irreps_out: IntoIrreps | None = None

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
        interaction_irreps = self._init_interaction_irreps(self.interaction_irreps, self.max_ell)

        # self._check_reachability(
        #     hidden_irreps=hidden_irreps,
        #     irreps_out=irreps_out,
        #     interaction_irreps=interaction_irreps,
        #     num_interactions=self.num_interactions,
        #     correlation_order=self.correlation_order,
        # )

        readout_mlp_irreps = (
            e3j.Irreps(self.readout_mlp_irreps) + non_scalar_irreps_out
        ).simplify()

        if self.num_features is None:
            num_features = int(functools.reduce(math.gcd, (mul for mul, _ in hidden_irreps)))
            hidden_irreps = e3j.Irreps([(mul // num_features, ir) for mul, ir in hidden_irreps])
        else:
            num_features = self.num_features

        # Build the layers we will use
        mace_layers = []
        readouts = []
        for i in range(self.num_interactions):
            is_not_first = i != 0
            is_not_last = i != self.num_interactions - 1

            # Mace
            mace_layer = MaceLayer(
                irreps_out,
                num_types=self.num_types,
                # Interaction
                num_features=num_features,
                interaction_irreps=interaction_irreps,
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
                # Globals
                global_interaction=self.global_interaction,
                global_correlation_order=self.global_correlation_order,
                global_irreps_out=self.global_irreps_out,
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
                    readout_mlp_irreps, irreps_out, activation=self.radial_activation
                )

            mace_layers.append(mace_layer)
            readouts.append(readout)

        self._layers = mace_layers
        self._readouts = readouts

    @jt.jaxtyped(typechecker=beartype.beartype)
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
                node_features=node_feats,
                node_types=node_types,
                # Edge features are not mutated, so just take directly from graph
                edge_features=graph.edges[keys.ATTRIBUTES],
                radial_embedding=graph.edges[keys.RADIAL_EMBEDDINGS],
                senders=graph.senders,
                receivers=graph.receivers,
                n_node=graph.n_node,
                global_features=graph.globals.get(keys.ATTRIBUTES),
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

    @jt.jaxtyped(typechecker=beartype.beartype)
    @staticmethod
    def _init_interaction_irreps(
        interaction_irreps: Literal["o3_restricted", "o3_full"] | IntoIrreps, ell_max: int
    ) -> e3j.Irreps:
        if interaction_irreps == "o3_restricted":
            return e3j.Irreps.spherical_harmonics(ell_max)

        if interaction_irreps == "o3_full":
            return e3j.Irreps(e3j.Irrep.iterator(ell_max))

        return e3j.Irreps(interaction_irreps)

    @staticmethod
    def _check_reachability(
        hidden_irreps: e3j.Irreps,
        irreps_out: e3j.Irreps,
        interaction_irreps: e3j.Irreps,
        num_interactions: int,
        correlation_order: int,
    ):
        """Simulates the tensor product paths to verify all requested irreps are physically
        reachable."""

        # Helper to do Irreps algebra but drop multiplicities (we only care if paths exist)
        def multiply_unique(ir1: e3j.Irreps, ir2: e3j.Irreps) -> e3j.Irreps:
            unique_irs = {ir for _, ir in e3j.tensor_product(ir1, ir2)}
            return e3j.Irreps([(1, ir) for ir in unique_irs])

        # Layer 0 node input is purely scalar
        reachable = e3j.Irreps("1x0e")

        for _ in range(num_interactions):
            # 1. Interaction Block: node features tensor-product with edge spherical harmonics
            messages = multiply_unique(reachable, interaction_irreps)

            # 2. Product Basis: tensor product of messages up to correlation_order
            layer_out_irs = {ir for _, ir in messages}
            current_power = messages

            for _ in range(1, correlation_order):
                current_power = multiply_unique(current_power, messages)
                layer_out_irs.update({ir for _, ir in current_power})

            reachable = e3j.Irreps([(1, ir) for ir in layer_out_irs])

        # Verification
        reachable_set = {ir for _, ir in reachable}

        # Check hidden_irreps (Warn: they just result in wasted parameters)
        unreachable_hidden = [str(ir) for _, ir in hidden_irreps if ir not in reachable_set]
        if unreachable_hidden:
            _LOGGER.warning(
                "MACE is configured with hidden_irreps containing %s, "
                "but given max_ell and correlation_order, these channels will never "
                "receive data and will remain zero throughout the network.",
                unreachable_hidden,
            )

        # Check irreps_out (Error: network cannot fulfill the physical target)
        unreachable_out = [str(ir) for _, ir in irreps_out if ir not in reachable_set]
        if unreachable_out:
            raise ValueError(
                f"The target irreps_out requires {unreachable_out}, but these are physically "
                f"unreachable after {num_interactions} layers. Increase max_ell, num_interactions, "
                f"or correlation_order."
            )
