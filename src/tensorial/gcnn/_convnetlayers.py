# -*- coding: utf-8 -*-
from typing import Dict, Union

import e3nn_jax as e3j
from flax import linen
import jraph

from . import _interaction_blocks, keys

__all__ = ('NequipLayer',)


class NequipLayer(linen.Module):
    """NequIP convolution layer.

    Implementation based on: https://github.com/mir-group/nequip/blob/main/nequip/nn/_convnetlayer.py
    """

    irreps_out: e3j.Irreps
    invariant_layers: int = 1
    invariant_neurons: int = 8
    # Radial
    radial_num_layers: int = 1
    radial_num_neurons: int = 8
    radial_activation: str = 'swish'

    avg_num_neighbours: int = 1.0
    activations: Union[str, Dict[str, str]] = _interaction_blocks.DEFAULT_ACTIVATIONS
    node_features_field = keys.FEATURES
    self_connection: bool = True
    num_species: int = 1

    resnet: bool = False

    def setup(self):
        self._interaction_block = _interaction_blocks.InteractionBlock(  # pylint: disable=attribute-defined-outside-init
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

    @linen.compact
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:  # pylint: disable=arguments-differ
        """
        # Apply a standard NequIP layer followed by an optional resnet step

        :param graph: the input graph
        :return: the output graph with node features updated
        """
        node_features = self._interaction_block(
            graph.nodes[keys.FEATURES], graph.edges[keys.ATTRIBUTES], graph.edges[keys.RADIAL_EMBEDDINGS],
            graph.senders, graph.receivers, graph.nodes.get(keys.SPECIES)
        )

        # If enabled, perform ResNet operation by adding back the old node features
        if self.resnet:
            node_features = node_features + graph.nodes[self.node_features_field]

        # Update the graph
        nodes = dict(graph.nodes)
        nodes[keys.FEATURES] = node_features
        return graph._replace(nodes=nodes)
