import beartype
import e3nn_jax as e3j
from flax import linen
import jax
import jax.numpy as jnp
import jaxtyping as jt
from jaxtyping import Bool, Float, Int
import reax.metrics

from tensorial import nn_utils
from tensorial.typing import Array, IndexArray, IntoIrreps, IrrepsArrayShape


class MessagePassingConvolution(linen.Module):
    irreps_out: IntoIrreps

    # Normalisation
    avg_num_neighbours: float | dict[int, float] | linen.FrozenDict[int, float] = 1.0
    epsilon: float | None = None

    # Radial
    radial_num_layers: int = 1
    radial_num_neurons: int = 8
    radial_activation: str | nn_utils.ActivationFunction = "swish"

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
        node_feats: IrrepsArrayShape["n_node node_irreps"],
        edge_features: IrrepsArrayShape["n_edge edge_irreps"],
        radial_embedding: Float[Array, "n_edge radial_embedding_dim"],
        senders: IndexArray["n_edge"],
        receivers: IndexArray["n_edge"],
        *,
        node_types: Int[Array, "n_node"] | None = None,
        edge_mask: Bool[Array, "n_edge"] | None = None,
    ) -> IrrepsArrayShape["n_node node_irreps_out"]:
        """
        Performs graph convolution operations on node and edge features using tensor products and
        radial embeddings.

        This function implements a message passing mechanism where node features are updated based
        on messages from neighbouring nodes and edge features. The messages are processed through a
        tensor product operation and weighted by a radial MLP that depends on inter-atomic
        distances.

        Args:
            node_feats: Node features with shape (n_nodes, node_irreps).
            edge_features: Edge features with shape (n_edges, edge_irreps).
            radial_embedding: Radial embeddings with shape (n_edges, radial_embedding_dim).
            senders: Sender indices for each edge with shape (n_edges,).
            receivers: Receiver indices for each edge with shape (n_edges,).
            edge_mask: Optional boolean mask for edges with shape (n_edges,).
            node_types: Optional node type indices with shape (n_nodes,).

        Returns:
            Updated node features with shape (n_nodes, node_irreps_out).

        Raises:
            ValueError: If the input shapes are inconsistent or invalid.
        """
        irreps_out = e3j.Irreps(self.irreps_out)  # Recast, because flax converts to tuple

        # The irreps to use for the output node features
        output_irreps = e3j.Irreps(self.irreps_out).regroup()

        messages = node_feats[senders]

        # Interaction between nodes and edges
        edge_features = e3j.tensor_product(
            messages, edge_features, filter_ir_out=output_irreps + "0e"
        )

        # Make a compound message
        messages: IrrepsArrayShape["n_edge node_irreps+edge_irreps"] = e3j.concatenate(
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

        if self.epsilon is not None:
            return node_feats * self.epsilon

        if isinstance(self.avg_num_neighbours, linen.FrozenDict):
            type_idxs = nn_utils.vwhere(node_types, self._types)
            avg_neighbours = self._avg_neighbours[type_idxs]
            return jax.vmap(lambda a, b: a / b)(node_feats, jnp.sqrt(avg_neighbours))

        return node_feats / jnp.sqrt(self.avg_num_neighbours)
