from typing import Optional, Union

import beartype
import e3nn_jax as e3j
from flax import linen
import jax.numpy as jnp
import jaxtyping as jt
import reax.metrics

from tensorial import nn_utils, typing


class MessagePassingConvolution(linen.Module):
    """Equivariant message passing convolution operation."""

    irreps_out: typing.IntoIrreps
    avg_num_neighbours: Union[float, dict[int, float]] = 1.0

    # Radial
    radial_num_layers: int = 1
    radial_num_neurons: int = 8
    radial_activation: Union[str, nn_utils.ActivationFunction] = "swish"

    def setup(self):
        # pylint: disable=attribute-defined-outside-init
        self._radial_act = nn_utils.get_jaxnn_activation(self.radial_activation)
        if isinstance(self.avg_num_neighbours, linen.FrozenDict):
            self._types = jnp.array(list(self.avg_num_neighbours.keys()))
            self._avg_neighbours = jnp.array(list(self.avg_num_neighbours.values()))

    @linen.compact
    @jt.jaxtyped(typechecker=beartype.beartype)
    def __call__(
        self,
        node_feats: typing.IrrepsArrayShape["n_nodes node_irreps"],
        edge_features: typing.IrrepsArrayShape["n_edges edge_irreps"],
        radial_embedding: jt.Float[typing.ArrayType, "n_edges radial_embedding_dim"],
        senders: typing.IndexArray["n_edges"],
        receivers: typing.IndexArray["n_edges"],
        *,
        edge_mask: Optional[jt.Bool[typing.ArrayType, "n_edges"]] = None,
        node_types: Optional[jt.Int[jt.Array, "n_nodes"]] = None,
    ) -> typing.IrrepsArrayShape["n_nodes node_irreps_out"]:
        irreps_out = e3j.Irreps(self.irreps_out)  # Recast, because flax converts to tuple

        # The irreps to use for the output node features
        output_irreps = e3j.Irreps(self.irreps_out).regroup()

        messages = node_feats[senders]

        # Interaction between nodes and edges
        edge_features = e3j.tensor_product(
            messages, edge_features, filter_ir_out=output_irreps + "0e"
        )

        # Make a compound message
        messages: typing.IrrepsArrayShape["n_edges node_irreps+edge_irreps"] = e3j.concatenate(
            [messages.filter(irreps_out + "0e"), edge_features]
        ).regroup()

        # Now, based on the messages irreps, create the radial MLP that maps from inter-atomic
        # distances to tensor product weights
        mlp = e3j.flax.MultiLayerPerceptron(
            (self.radial_num_neurons,) * self.radial_num_layers + (messages.irreps.num_irreps,),
            self._radial_act,
            with_bias=False,  # do not use bias so that R(0) = 0
            output_activation=False,
        )
        # Get weights for the tensor product from our full-connected MLP
        if edge_mask is not None:
            radial_embedding = jnp.where(
                reax.metrics.utils.prepare_mask(radial_embedding, edge_mask),
                radial_embedding,
                0.0,
            )
        weights = mlp(radial_embedding)
        if edge_mask is not None:
            weights = jnp.where(
                reax.metrics.utils.prepare_mask(radial_embedding, edge_mask), weights, 0.0
            )
        messages = messages * weights

        zeros = e3j.zeros(messages.irreps, node_feats.shape[:1], messages.dtype)
        node_feats = zeros.at[receivers].add(messages)

        if isinstance(self.avg_num_neighbours, linen.FrozenDict):
            type_idxs = nn_utils.vwhere(node_types, self._types)
            avg_neighbours = self._avg_neighbours[type_idxs]
            return node_feats / jnp.sqrt(avg_neighbours).reshape(-1, 1)

        return node_feats / jnp.sqrt(self.avg_num_neighbours)
