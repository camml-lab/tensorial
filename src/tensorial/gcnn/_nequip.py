from collections.abc import Callable, Mapping
import functools
from typing import Optional, Union

import beartype
import e3nn_jax as e3j
from flax import linen
import jaxtyping as jt
import jraph

from . import _base, _message_passing, keys
from .. import nn_utils, typing
from .. import utils as tensorial_utils

__all__ = ("NequipLayer",)

# Default activations used by gate
DEFAULT_ACTIVATIONS = linen.FrozenDict({"e": "silu", "o": "tanh"})
ActivationLike = Union[str, nn_utils.ActivationFunction]


class InteractionBlock(linen.Module):
    """NequIP style interaction block.

    Implementation based on
        https://github.com/mir-group/nequip/blob/main/nequip/nn/_interaction_block.py
    and
        https://github.com/mariogeiger/nequip-jax/blob/main/nequip_jax/nequip.py

    :param irreps_out: the irreps of the output node features
    :param radial_num_layers: the number of layers in the radial MLP
    :param radial_num_neurons: the number of neurons per layer in the radial MLP
    :param radial_activation: activation function used by radial MLP
    :param avg_num_neighbours: average number of neighbours of each node, used for normalisation
    :param skip_connection: If True, skip connection will be applied at end of interaction
    """

    irreps_out: typing.IntoIrreps = 4 * e3j.Irreps("0e + 1o + 2e")
    # Radial
    radial_num_layers: int = 1
    radial_num_neurons: int = 8
    radial_activation: ActivationLike = "swish"

    avg_num_neighbours: Union[float, dict[int, float]] = 1.0
    skip_connection: bool = True
    activations: Union[str, Mapping[str, ActivationLike]] = DEFAULT_ACTIVATIONS

    num_species: int = 1

    def setup(self):
        # pylint: disable=attribute-defined-outside-init
        self._message_passing = _message_passing.MessagePassingConvolution(
            irreps_out=self.irreps_out,
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
        node_features: typing.IrrepsArrayShape["n_nodes irreps"],
        edge_features: typing.IrrepsArrayShape["n_edges edge_irreps"],
        radial_embedding: jt.Float[typing.ArrayType, "n_edges radial_embedding_dim"],
        senders: typing.IndexArray["n_edges"],
        receivers: typing.IndexArray["n_edges"],
        node_species: Optional[jt.Int[typing.ArrayType, "n_nodes"]] = None,
        *,
        node_mask: Optional[jt.Bool[typing.ArrayType, "n_nodes"]] = None,
        edge_mask: Optional[jt.Bool[typing.ArrayType, "n_edges"]] = None,
    ) -> e3j.IrrepsArray:
        """
        A NequIP interaction made up of the following steps:

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

        if node_mask is not None:
            node_features = e3j.where(
                node_mask, node_features, tensorial_utils.zeros_like(node_features)
            )

        node_feats = e3j.flax.Linear(node_features.irreps, name="linear_up")(node_features)
        if node_mask is not None:
            node_features = e3j.where(
                node_mask, node_features, tensorial_utils.zeros_like(node_features)
            )

        node_feats = self._message_passing(
            node_feats, edge_features, radial_embedding, senders, receivers, edge_mask=edge_mask
        )

        gate_irreps = output_irreps.filter(keep=node_feats.irreps)
        num_non_scalar = gate_irreps.filter(drop="0e + 0o").num_irreps
        gate_irreps = gate_irreps + (num_non_scalar * e3j.Irrep("0e"))

        # Second linear, now we create any extra gate scalars
        node_feats = e3j.flax.Linear(gate_irreps, name="linear_down")(node_feats)

        # self-connection: species weighted tensor product that maps to current irreps space
        if self.skip_connection:
            skip = e3j.flax.Linear(
                node_feats.irreps,
                num_indexed_weights=self.num_species,
                name="skip_connection",
                force_irreps_out=True,
            )(node_species, node_features)
            node_feats = 0.5 * (node_feats + skip)

        # Apply non-linearity
        node_feats = self._gate(node_feats)
        return node_feats


class NequipLayer(linen.Module):
    """
    NequIP convolution layer.

    Implementation based on:
    https://github.com/mir-group/nequip/blob/main/nequip/nn/_convnetlayer.py
    """

    irreps_out: typing.IntoIrreps
    invariant_layers: int = 1
    invariant_neurons: int = 8
    # Radial
    radial_num_layers: int = 1
    radial_num_neurons: int = 8
    radial_activation: ActivationLike = "swish"

    avg_num_neighbours: Union[float, dict[int, float]] = 1.0
    activations: Union[str, Mapping[str, ActivationLike]] = DEFAULT_ACTIVATIONS
    node_features_field = keys.FEATURES
    skip_connection: bool = True
    num_species: int = 1

    interaction_block: Callable = None

    resnet: bool = False

    def setup(self):
        # pylint: disable=attribute-defined-outside-init
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
            )
        else:
            self._interaction_block = self.interaction_block

    @linen.compact
    @jt.jaxtyped(typechecker=beartype.beartype)
    @_base.shape_check
    def __call__(
        self, graph: jraph.GraphsTuple
    ) -> jraph.GraphsTuple:  # pylint: disable=arguments-differ
        """
        Apply a standard NequIP layer followed by an optional resnet step

        :param graph: the input graph
        :return: the output graph with node features updated
        """
        node_features = self._interaction_block(
            graph.nodes[keys.FEATURES],
            graph.edges[keys.ATTRIBUTES],
            graph.edges[keys.RADIAL_EMBEDDINGS],
            graph.senders,
            graph.receivers,
            graph.nodes.get(keys.SPECIES),
            node_mask=graph.nodes.get(keys.MASK),
            edge_mask=graph.edges.get(keys.MASK),
        )

        # If enabled, perform ResNet operation by adding back the old node features
        if self.resnet:
            node_features = node_features + graph.nodes[self.node_features_field]

        # Update the graph
        nodes = dict(graph.nodes)
        nodes[keys.FEATURES] = node_features
        return graph._replace(nodes=nodes)
