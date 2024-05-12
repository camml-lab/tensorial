# -*- coding: utf-8 -*-
from typing import Optional

import e3nn_jax as e3j
from flax import linen
import jax
import jax.numpy as jnp

from tensorial import nn_utils


class MessagePassingConvolution(linen.Module):
    irreps_out: e3j.Irreps
    avg_num_neighbors: float = 1.0

    # Radial
    radial_activation: nn_utils.ActivationFunction
    radial_num_layers: int = 1
    radial_num_neurons: int = 8

    def setup(self):
        # pylint: disable=attribute-defined-outside-init
        self._radial_act = nn_utils.get_jaxnn_activation(self.radial_activation)

    def __call__(
        self,
        node_feats: e3j.IrrepsArray,  # [n_nodes, node_irreps]
        edge_features: e3j.IrrepsArray,  # [n_edges, edge_irreps]
        radial_embedding: jnp.ndarray,  # [n_edges, radial_embedding_dim]
        senders: jax.Array,  # [n_edges]
        receivers: jax.Array,  # [n_edges]
        edge_mask: Optional[jax.Array] = None,
    ) -> e3j.IrrepsArray:
        assert node_feats.ndim == 2

        # The irreps to use for the output node features
        output_irreps = e3j.Irreps(self.irreps_out).regroup()

        messages = node_feats[senders]

        # Interaction between nodes and edges
        edge_features = e3j.tensor_product(
            messages, edge_features, filter_ir_out=output_irreps + "0e"
        )

        # Make a compound message [n_edges, node_irreps + edge_irreps]
        messages = e3j.concatenate(
            [messages.filter(self.irreps_out + "0e"), edge_features]
        ).regroup()

        # Now, based on the messages irreps, create the radial MLP that maps from inter-atomic
        # distances to tensor product weights
        mlp = e3j.flax.MultiLayerPerceptron(
            (self.radial_num_neurons,) * self.radial_num_layers + (messages.irreps.num_irreps,),
            nn_utils.get_jaxnn_activation(self._radial_act),
            with_bias=False,  # do not use bias so that R(0) = 0
            output_activation=False,
        )
        # Get weights for the tensor product from our full-connected MLP
        if edge_mask is not None:
            radial_embedding = jnp.where(
                nn_utils.prepare_mask(edge_mask, radial_embedding),
                radial_embedding,
                0.0,
            )
        weights = mlp(radial_embedding)
        if edge_mask is not None:
            weights = jnp.where(nn_utils.prepare_mask(edge_mask, radial_embedding), weights, 0.0)
        messages = messages * weights  # [n_edges, message irreps]

        zeros = e3j.zeros(messages.irreps, node_feats.shape[:1], messages.dtype)
        node_feats = zeros.at[receivers].add(messages)  # [n_nodes, node irreps]

        return node_feats / jnp.sqrt(self.avg_num_neighbors)
