# -*- coding: utf-8 -*-
import functools
from typing import Callable, Dict, Optional, Union

import beartype
import e3nn_jax as e3j
from flax import linen
import jax
import jaxtyping as jt
import jraph

from tensorial import nn_utils

from . import _message_passing, keys

__all__ = ("NequipLayer",)


# Default activations used by gate
DEFAULT_ACTIVATIONS = linen.FrozenDict({"e": "silu", "o": "tanh"})


@jt.jaxtyped(beartype.beartype)
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
    :param self_connection: If True, self connection will be applied at end of interaction
    """

    irreps_out: e3j.Irreps = 4 * e3j.Irreps("0e + 1o + 2e")
    # Radial
    radial_num_layers: int = 1
    radial_num_neurons: int = 8
    radial_activation: nn_utils.ActivationFunction = "swish"

    avg_num_neighbours: float = 1.0
    self_connection: bool = True
    activations: Union[str, Dict[str, nn_utils.ActivationFunction]] = DEFAULT_ACTIVATIONS

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
    def __call__(
        self,
        node_features: jt.Float[e3j.IrrepsArray, "n_nodes irreps"],
        edge_features: jt.Float[e3j.IrrepsArray, "n_edges irreps"],
        radial_embedding: jt.Float[jax.Array, "n_edges radial_embedding_dim"],
        senders: jt.Int[jax.Array, "n_edges"],
        receivers: jt.Int[jax.Array, "n_edges"],
        node_species: Optional[jt.Int[jax.Array, "n_nodes"]] = None,
        edge_mask: Optional[jt.Bool[jax.Array, "n_edges"]] = None,
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

        node_feats = e3j.flax.Linear(node_features.irreps, name="linear_up")(node_features)

        node_feats = self._message_passing(
            node_feats, edge_features, radial_embedding, senders, receivers, edge_mask=edge_mask
        )

        gate_irreps = output_irreps.filter(keep=node_feats.irreps)
        num_non_scalar = gate_irreps.filter(drop="0e + 0o").num_irreps
        gate_irreps = gate_irreps + (num_non_scalar * e3j.Irrep("0e"))

        # Second linear, now we create any extra gate scalars
        node_feats = e3j.flax.Linear(gate_irreps, name="linear_down")(node_feats)

        # self-connection: species weighted tensor product that maps to current irreps space
        if self.self_connection:
            skip = e3j.flax.Linear(
                node_feats.irreps,
                num_indexed_weights=self.num_species,
                name="self_connection",
                force_irreps_out=True,
            )(node_species, node_features)
            node_feats = 0.5 * (node_feats + skip)

        # Apply non-linearity
        node_feats = self._gate(node_feats)
        return node_feats


@jt.jaxtyped(beartype.beartype)
class NequipLayer(linen.Module):
    """
    NequIP convolution layer.

    Implementation based on:
    https://github.com/mir-group/nequip/blob/main/nequip/nn/_convnetlayer.py
    """

    irreps_out: e3j.Irreps
    invariant_layers: int = 1
    invariant_neurons: int = 8
    # Radial
    radial_num_layers: int = 1
    radial_num_neurons: int = 8
    radial_activation: str = "swish"

    avg_num_neighbours: int = 1.0
    activations: Union[str, Dict[str, str]] = DEFAULT_ACTIVATIONS
    node_features_field = keys.FEATURES
    self_connection: bool = True
    num_species: int = 1

    interaction_block: Callable = None

    resnet: bool = False

    def setup(self):
        # pylint: disable=attribute-defined-outside-init
        if self.interaction_block is None:
            self._interaction_block = InteractionBlock(
                self.irreps_out,
                # Radial
                radial_num_layers=self.radial_num_layers,
                radial_num_neurons=self.radial_num_neurons,
                radial_activation=self.radial_activation,
                avg_num_neighbours=self.avg_num_neighbours,
                self_connection=self.self_connection,
                activations=self.activations,
                num_species=self.num_species,
            )
        else:
            self._interaction_block = self.interaction_block

    @linen.compact
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
            edge_mask=graph.edges.get(keys.MASK, None),
        )

        # If enabled, perform ResNet operation by adding back the old node features
        if self.resnet:
            node_features = node_features + graph.nodes[self.node_features_field]

        # Update the graph
        nodes = dict(graph.nodes)
        nodes[keys.FEATURES] = node_features
        return graph._replace(nodes=nodes)
