"""NequIP-style equivariant graph neural network layers.

Provides `InteractionBlock`, `NequipLayer` (a full convolution layer wrapping
an interaction block plus invariant layers) and `Nequip` (a stack of `NequipLayer` layers),
a faithful port of the NEquIP model commonly used for force-field regression.
"""

from collections.abc import Callable, Mapping
import functools
import math
from typing import cast

import beartype
import e3nn_jax as e3j
from flax import linen
import jaxtyping as jt
from jaxtyping import Bool, Float, Int
import jraph

from tensorial.typing import Array, IndexArray, IntoIrreps, IrrepsArrayShape

from .. import nn_utils
from .. import utils as tensorial_utils
from . import _base, _message_passing, keys

__all__ = "NequipLayer", "Nequip"

# Default activations used by gate
DEFAULT_ACTIVATIONS = linen.FrozenDict({"e": "silu", "o": "tanh"})
ActivationLike = str | nn_utils.ActivationFunction


def _condition(
    features: e3j.IrrepsArray, attributes: e3j.IrrepsArray, irreps_out: e3j.Irreps
) -> e3j.IrrepsArray:
    """Tensor product of node features with node attributes, keeping only the irreps in
    `irreps_out`.

    The result always feeds an `e3j.flax.Linear` to `irreps_out`, which has no path from any
    other irrep, so filtering here gives the same output (each path is normalised on its own)
    while skipping work.  This matters for non-scalar attributes (e.g. an external field),
    where the full product contains many irreps the linear would discard.
    """
    return e3j.tensor_product(features, attributes, filter_ir_out=irreps_out)


class InteractionBlock(linen.Module):
    """NequIP style interaction block.

    Implementation based on
        https://github.com/mir-group/nequip/blob/main/nequip/nn/_interaction_block.py
    and
        https://github.com/mariogeiger/nequip-jax/blob/main/nequip_jax/nequip.py

    Args:
        irreps_out: the irreps of the output node features
        radial_num_layers: the number of layers in the radial MLP
        radial_num_neurons: the number of neurons per layer in the
            radial MLP
        radial_activation: activation function used by radial MLP
        avg_num_neighbours: average number of neighbours of each node,
            used for normalization
        skip_connection: If True, skip connection will be applied at end
            of interaction
        attributes_in_linears: with ``node_attributes``, also condition the linears before and
            after the convolution on them (as e3nn's ``gate_points_2101`` convolution does), not
            only the self-connection

    Node attributes: by default the self-connection has one weight set per species
    (``node_species``). Passing continuous ``node_attributes`` (scalars, e.g. a species one-hot
    concatenated with per-node descriptors) makes it the tensor product of the node features
    with them followed by a linear, as in the original NequIP; for a one-hot this is the same
    function family as the per-species weights.
    """

    irreps_out: IntoIrreps = 4 * e3j.Irreps("0e + 1o + 2e")
    # Radial
    radial_num_layers: int = 1
    radial_num_neurons: int = 8
    radial_activation: ActivationLike = "swish"

    avg_num_neighbours: float | dict[int, float] = 1.0
    skip_connection: bool = True
    activations: str | Mapping[str, ActivationLike] = DEFAULT_ACTIVATIONS

    num_species: int = 1
    attributes_in_linears: bool = False

    def setup(self):
        """Build the underlying `MessagePassingConvolution` for this block."""
        self._message_passing = _message_passing.MessagePassingConvolution(
            self.irreps_out,
            avg_num_neighbours=self.avg_num_neighbours,
            radial_num_layers=self.radial_num_layers,
            radial_num_neurons=self.radial_num_neurons,
            radial_activation=self.radial_activation,
        )

        self._gate = functools.partial(
            e3j.gate,
            even_act=nn_utils.get_jaxnn_activation(self.activations["e"]),
            odd_act=nn_utils.get_jaxnn_activation(self.activations["o"]),
            even_gate_act=nn_utils.get_jaxnn_activation(self.activations["e"]),
            odd_gate_act=nn_utils.get_jaxnn_activation(self.activations["o"]),
        )
        self._radial_act = nn_utils.get_jaxnn_activation(self.radial_activation)

    @linen.compact
    @jt.jaxtyped(typechecker=beartype.beartype)
    def __call__(
        self,
        node_features: IrrepsArrayShape["n_node irreps"],
        edge_features: IrrepsArrayShape["n_edge edge_irreps"],
        radial_embedding: Float[Array, "n_edge radial_embedding_dim"],
        senders: IndexArray["n_edge"],
        receivers: IndexArray["n_edge"],
        *,
        node_species: Int[Array, "n_node"] | None = None,
        node_attributes: IrrepsArrayShape["n_node attr_irreps"] | None = None,
        node_mask: Bool[Array, "n_node"] | None = None,
        edge_mask: Bool[Array, "n_edge"] | None = None,
    ) -> e3j.IrrepsArray:
        """A NequIP interaction made up of the following steps:

        - linear on nodes
        - tensor product + aggregate
        - divide by sqrt(average number of neighbors)
        - concatenate
        - linear on nodes
        - gate non-linearity
        """
        # The irreps to use for the output node features
        output_irreps = e3j.Irreps(self.irreps_out).regroup()
        if node_mask is not None:
            node_mask = nn_utils.prepare_mask(node_mask, node_features)
            node_features = e3j.where(
                node_mask, node_features, tensorial_utils.zeros_like(node_features)
            )

        conditioned = node_attributes is not None and self.attributes_in_linears
        # No bias, so masked (zeroed) nodes stay zero and need no re-masking
        node_feats = e3j.flax.Linear(node_features.irreps, name="linear_up")(
            _condition(node_features, node_attributes, node_features.irreps)
            if conditioned
            else node_features
        )

        node_feats = self._message_passing(
            node_feats,
            edge_features,
            radial_embedding,
            senders,
            receivers,
            node_types=node_species,
            edge_mask=edge_mask,
        )

        gate_irreps = output_irreps.filter(keep=node_feats.irreps)
        num_non_scalar = gate_irreps.filter(drop="0e + 0o").num_irreps
        gate_irreps = gate_irreps + (num_non_scalar * e3j.Irrep("0e"))

        # Second linear, now we create any extra gate scalars
        node_feats = e3j.flax.Linear(gate_irreps, name="linear_down")(
            _condition(node_feats, node_attributes, gate_irreps) if conditioned else node_feats
        )

        # self-connection: species weighted tensor product that maps to current irreps space
        if self.skip_connection:
            if node_attributes is None:
                skip = e3j.flax.Linear(
                    node_feats.irreps,
                    num_indexed_weights=self.num_species,
                    name="skip_connection",
                    force_irreps_out=True,
                )(node_species, node_features)
            else:
                # Tensor product with the (continuous) node attributes, then a linear
                skip = e3j.flax.Linear(
                    node_feats.irreps, name="skip_connection", force_irreps_out=True
                )(_condition(node_features, node_attributes, node_feats.irreps))
            # Scale by 1/sqrt(2) so that the sum of two (roughly independent, at initialisation)
            # branches keeps the same variance as when the self-connection is disabled
            node_feats = (node_feats + skip) / math.sqrt(2.0)

        # Apply non-linearity
        node_feats = self._gate(node_feats)
        return node_feats


class NequipLayer(linen.Module):
    """NequIP convolution layer.

    Implementation based on:
    https://github.com/mir-group/nequip/blob/main/nequip/nn/_convnetlayer.py
    """

    irreps_out: IntoIrreps
    invariant_layers: int = 1
    invariant_neurons: int = 8
    # Radial
    radial_num_layers: int = 1
    radial_num_neurons: int = 8
    radial_activation: ActivationLike = "swish"

    avg_num_neighbours: float | dict[int, float] = 1.0
    activations: str | Mapping[str, ActivationLike] = DEFAULT_ACTIVATIONS
    node_features_field = keys.FEATURES
    skip_connection: bool = True
    num_species: int = 1
    # Continuous node attributes for the self-connection (see InteractionBlock), e.g. "attributes"
    node_attributes_field: str | None = None
    attributes_in_linears: bool = False

    interaction_block: Callable | None = None

    resnet: bool = False

    def setup(self):
        """Build the interaction block (and invariant layers) for the NequIP layer."""
        if self.interaction_block is None:
            self._interaction_block = InteractionBlock(
                self.irreps_out,
                radial_num_layers=self.radial_num_layers,
                radial_num_neurons=self.radial_num_neurons,
                radial_activation=self.radial_activation,
                avg_num_neighbours=self.avg_num_neighbours,
                skip_connection=self.skip_connection,
                activations=self.activations,
                num_species=self.num_species,
                attributes_in_linears=self.attributes_in_linears,
            )
        else:
            self._interaction_block = self.interaction_block

    @linen.compact
    @jt.jaxtyped(typechecker=beartype.beartype)
    @_base.shape_check
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """Apply a standard NequIP layer followed by an optional resnet step

        Args:
            graph: the input graph

        Returns:
            the output graph with node features updated
        """
        nodes = dict(cast("dict[str, Any]", graph.nodes))
        edges = cast("dict[str, Any]", graph.edges)
        species = nodes.get(keys.SPECIES)
        node_species = species[:, 0] if species is not None else None

        node_features = self._interaction_block(
            nodes[keys.FEATURES],
            edges[keys.ATTRIBUTES],
            edges[keys.RADIAL_EMBEDDINGS],
            graph.senders,
            graph.receivers,
            node_species=node_species,
            node_attributes=(
                None if self.node_attributes_field is None else nodes[self.node_attributes_field]
            ),
            node_mask=nodes.get(keys.MASK),
            edge_mask=edges.get(keys.MASK),
        )

        # If enabled, perform ResNet operation by adding back the old node features.  As in
        # NequIP, this is only possible (and so only done) when the irreps are unchanged
        if self.resnet:
            prev_features = nodes[self.node_features_field]
            if prev_features.irreps == node_features.irreps:
                node_features = node_features + prev_features

        # Update the graph
        nodes[keys.FEATURES] = node_features
        return graph._replace(nodes=nodes)


class Nequip(linen.Module):
    """NequIP equivariant graph neural network: a stack of :class:`NequipLayer` interactions.

    Every layer maps the node features to ``hidden_irreps``, so the whole stack is described
    by a handful of numbers instead of one config entry per layer.  As with
    :class:`tensorial.gcnn.Mace`, the embeddings that come before (species, edge spherical
    harmonics, radial basis, initial ``NodewiseLinear``) and any readout that comes after are
    left to the surrounding model, which keeps this module independent of the task.

    The module expects the graph to contain, at minimum:

    * ``nodes.features`` — the node features (an ``e3nn_jax.IrrepsArray``)
    * ``edges.attributes`` — the edge features (an ``e3nn_jax.IrrepsArray``)
    * ``edges.radial_embeddings`` — radial edge embeddings, shape ``[n_edge, D]``
    * ``nodes.species`` — integer node types, shape ``[n_node, 1]``, used when
      ``num_species > 1``

    and writes the updated node features back to ``nodes.features``.

    Args:
        hidden_irreps: the irreps of the node features produced by every layer, e.g.
            ``make_irreps(mul=16, ell_max=2)``
        num_layers: the number of interaction layers
        num_species: the number of node types.  With more than one, the self-connection of
            every layer has separate weights per type
        avg_num_neighbours: average number of neighbours of each node, used for
            normalisation.  Can be a single value or a mapping of node type to value
        radial_num_layers: the number of layers in the radial MLP
        radial_num_neurons: the number of neurons per layer in the radial MLP
        radial_activation: activation function used by the radial MLP
        activations: the gate activations for even (``"e"``) and odd (``"o"``) scalars
        skip_connection: if ``True``, every interaction adds a self-connection
        skip_connection_first_layer: if ``False``, the first layer has no self-connection
            (even when ``skip_connection`` is ``True``).  This is what NequIP does by default:
            an isolated atom (no neighbours) then has zero features after the first layer, and
            so after every layer, which leaves its energy to the per-type shift of the readout
        resnet: if ``True``, every layer adds its input node features to its output, provided
            the irreps are the same (layers whose irreps change, typically the first, skip it)
        node_attributes_field: if given, the node field with continuous (scalar) attributes that
            the self-connections are conditioned on through a tensor product, as in the original
            NequIP, instead of per-species weights; e.g. a species one-hot together with
            per-node descriptors
        attributes_in_linears: also condition the linears around each convolution on them

    Example:
        >>> model = Nequip(
        ...     hidden_irreps=make_irreps(mul=16, ell_max=2),
        ...     num_layers=3,
        ...     num_species=4,
        ...     avg_num_neighbours=12.0,
        ... )
    """

    hidden_irreps: IntoIrreps
    num_layers: int = 3
    num_species: int = 1
    avg_num_neighbours: float | Mapping[int, float] = 1.0
    # Radial
    radial_num_layers: int = 1
    radial_num_neurons: int = 8
    radial_activation: ActivationLike = "swish"

    activations: str | Mapping[str, ActivationLike] = DEFAULT_ACTIVATIONS
    skip_connection: bool = True
    skip_connection_first_layer: bool = True
    resnet: bool = False
    node_attributes_field: str | None = None
    attributes_in_linears: bool = False

    def setup(self):
        """Build the stack of interaction layers."""
        if self.num_layers < 1:
            raise ValueError(f"'num_layers' must be at least 1, got {self.num_layers}")

        self._layers = [
            NequipLayer(
                self.hidden_irreps,
                radial_num_layers=self.radial_num_layers,
                radial_num_neurons=self.radial_num_neurons,
                radial_activation=self.radial_activation,
                avg_num_neighbours=self.avg_num_neighbours,
                activations=self.activations,
                skip_connection=self.skip_connection
                and (i > 0 or self.skip_connection_first_layer),
                num_species=self.num_species,
                resnet=self.resnet,
                node_attributes_field=self.node_attributes_field,
                attributes_in_linears=self.attributes_in_linears,
            )
            for i in range(self.num_layers)
        ]

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """Apply every interaction layer in turn.

        Args:
            graph: the input graph

        Returns:
            the output graph with node features updated
        """
        for layer in self._layers:
            graph = layer(graph)

        return graph
