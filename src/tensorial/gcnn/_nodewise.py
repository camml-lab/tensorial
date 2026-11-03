import logging
from typing import TYPE_CHECKING, cast

import e3nn_jax as e3j
from flax import linen
import jax.numpy as jnp
import jraph

from .. import base
from ..tensors import SphericalHarmonic
from . import _base, _tree, graph_ops, keys, utils
from .experimental import utils as exp_utils

if TYPE_CHECKING:
    import tensorial

__all__ = (
    "NodewiseLinear",
    "NodewiseReduce",
    "NodewiseEmbedding",
    "NodewiseEncoding",
    "NodewiseDecoding",
    "AllPairsLongRangeEncoding",
)

_LOGGER = logging.getLogger(__name__)


class NodewiseLinear(linen.Module):
    """Nodewise linear operation"""

    irreps_out: str | e3j.Irreps
    irreps_in: e3j.Irreps | None = None
    field: str = keys.FEATURES
    out_field: str | None = keys.FEATURES
    num_types: int | None = None
    types_field: str | None = None

    def setup(self):
        # pylint: disable=attribute-defined-outside-init
        self.linear = e3j.flax.Linear(
            irreps_out=self.irreps_out,
            irreps_in=self.irreps_in,
            num_indexed_weights=self.num_types,
            force_irreps_out=True,
        )
        # Set the default of the types field if num_types is supplied and the user didn't supply it
        if self.types_field is None:
            self._types_field = keys.SPECIES if self.num_types else None
        else:
            if not self.num_types:
                _LOGGER.warning(
                    "User supplied a ``types_field``, %s, but failed to supply ``num_types``. "
                    "Ignoring.",
                    self._types_field,
                )
            self._types_field = self.types_field

    @_base.shape_check
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        nodes = cast("dict[str, Any]", graph.nodes)
        if self.num_types:
            # We are using weights indexed by the type
            features = self.linear(nodes[self._types_field][:, 0], nodes[self.field])
        else:
            features = self.linear(nodes[self.field])

        nodes[self.out_field] = features
        return graph._replace(nodes=nodes)


class NodewiseReduce(linen.Module):
    """Applies a reduction operation over node features and stores the result in the graph globals.

    This module reduces a specified field in the graph's node features across all nodes
    (within each graph if batched) using a specified reduction operation (`sum`, `mean`,
    or `normalized_sum`). The result is written to the `globals` field of the `GraphsTuple`.

    Attributes:
        field (str): Path to the node field to reduce, e.g. "energy" or "features.energy".
        out_field (Optional[str]): Path to the output global field. If None, defaults to
            "<reduce>_<field>" under `globals`.
        reduce (str): Reduction operation to apply. Must be one of "sum", "mean", or
            "normalized_sum".
        average_num_atoms (float): Required if `reduce` is "normalized_sum". Used to scale the
            result by `average_num_atoms ** -0.5`.

    Raises:
        ValueError: If `reduce` is not one of the allowed options.
        ValueError: If `reduce == "normalized_sum"` but `average_num_atoms` is not provided.

    Returns:
        A new `GraphsTuple` with the reduced value written to the specified `globals` field.
    """

    field: str
    out_field: str | None = None
    reduce: str = "sum"
    average_num_atoms: float = None
    as_array: bool = False

    def setup(self):
        # pylint: disable=attribute-defined-outside-init
        if self.reduce not in ("sum", "mean", "normalized_sum"):
            raise ValueError(self.reduce)

        self._field = ("nodes",) + utils.path_from_str(
            self.field if self.field is not None else tuple()
        )
        self._out_field = ("globals",) + utils.path_from_str(
            self.out_field or f"{self.reduce}_{self.field}"
        )

        if self.reduce == "normalized_sum":
            if self.average_num_atoms is None:
                raise ValueError(self.average_num_atoms)
            self.constant = float(self.average_num_atoms) ** -0.5
            self._reduce = "sum"
        else:
            self.constant = 1.0
            self._reduce = self.reduce

    @_base.shape_check
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        reduced = self.constant * graph_ops.graph_segment_reduce(graph, self._field, self._reduce)
        if self.as_array and isinstance(reduced, e3j.IrrepsArray):
            reduced = reduced.array

        return exp_utils.update_graph(graph).set(self._out_field, reduced).get()


class NodewiseEmbedding(linen.Module):
    """Take the attributes in the nodes dictionary given by attrs, embed them, and store the results
    as a direct sum of irreps in the out_field.
    """

    attrs: "tensorial.IrrepsTree"
    out_field: str = keys.ATTRIBUTES
    node_shape_from: str | None = keys.POSITIONS

    @_base.shape_check
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        nodes = cast("dict[str, Any]", graph.nodes)
        if isinstance(self.attrs, (dict, linen.FrozenDict)):
            values = []
            for key, attr in self.attrs.items():
                path = _tree.path_from_str(key)
                if len(path) > 1:
                    if path[0] not in ("nodes", "globals"):
                        raise ValueError(f"The attribute key must not contain '.', got: {key}")
                else:
                    # Assume it is a node attribute
                    path = ("nodes",) + path

                try:
                    value = _tree.get(graph, path)
                except KeyError:
                    root = path[0]
                    raise ValueError(
                        f"Did not find '{_tree.path_to_str(path[1:])}' in "
                        f"{root}, available entries are: "
                        f"{', '.join(_tree.get(graph, root).keys())}"
                    ) from None

                value: e3j.IrrepsArray = base.create_tensor(attr, value)

                if path[0] == "globals":
                    if self.node_shape_from is None:
                        # This will not have a static shape, so will cause recompilation
                        total_repeat_length = jnp.sum(graph.n_node)
                    else:
                        total_repeat_length = nodes[self.node_shape_from].shape[0]

                    if len(value.shape) < 2:
                        value = value.broadcast_to((total_repeat_length, *value.shape))
                    else:
                        array = jnp.repeat(
                            value.array,
                            graph.n_node,
                            axis=0,
                            total_repeat_length=total_repeat_length,
                        )
                        value = e3j.IrrepsArray(value.irreps, array)

                values.append(value)

            encoded: e3j.IrrepsArray = e3j.concatenate(values)
        else:
            values = nodes
            # Create the embedding
            encoded: e3j.IrrepsArray = base.create_tensor(self.attrs, values)

        # Store in output field
        nodes[self.out_field] = encoded
        return graph._replace(nodes=nodes)


class NodewiseDecoding(linen.Module):
    """Decode the direct sum of irreps stored in the in_field and store each tensor as a node value
    with key coming from the attrs.
    """

    attrs: "tensorial.IrrepsTree"
    in_field: str = keys.ATTRIBUTES

    @_base.shape_check
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        # Here, we need to split up the direct sum of irreps in the in field, and save the values
        # in the nodes dict corresponding to the attrs keys
        idx = 0
        nodes_dict = cast("dict[str, Any]", graph.nodes)
        irreps_tensor = nodes_dict[self.in_field]
        for key, value in base.tensorial_attrs(self.attrs).items():
            irreps = base.irreps(value)
            tensor_slice = irreps_tensor[..., idx : idx + irreps.dim]
            nodes_dict[key] = base.from_tensor(value, tensor_slice)
            idx += irreps.dim

        # All done, return the new graph
        return graph._replace(nodes=nodes_dict)


# For legacy reasons
NodewiseEncoding = NodewiseEmbedding


class AllPairsLongRangeEncoding(linen.Module):
    """Dense all-pairs node encoding.

    For each centre atom ``j``, computes

        M_j^(l,m) = sum_{i != j} r_ij^l Y_lm(r_hat_ij),

    over all valid source atoms ``i`` in the same graph, with

        r_ij = x_j - x_i.

    The encoding is not restricted by the short-range cutoff graph.
    """

    l_max: int
    field: str = keys.ATTRIBUTES
    normalise: bool = True
    time_field: str = "time_embedding"

    @_base.shape_check
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        positions = graph.nodes[keys.POSITIONS]
        num_nodes = positions.shape[0]

        # Follow the convention used by _spatial.with_edge_vectors:
        # `edge_vecs = pos[graph.receivers] - pos[graph.senders]`;
        # reflecting the message passing convention in `MessagePassingConvolution`:
        # `messages = node_feats[senders]` and `zeros.at[receivers].add(messages)`.
        # Here j is the receiver/centre and i is the sender/source.
        pair_vectors = positions[:, None, :] - positions[None, :, :]
        pair_distances = jnp.linalg.norm(pair_vectors, axis=-1)

        # Identify the graph each node belongs to.
        graph_ids = jnp.repeat(
            jnp.arange(len(graph.n_node)),
            graph.n_node,
            total_repeat_length=num_nodes,
        )

        valid_nodes = graph.nodes[keys.MASK]

        # Keep distinct, valid nodes belonging to the same graph.
        node_ids = jnp.arange(num_nodes)
        same_graph = graph_ids[:, None] == graph_ids[None, :]
        different_nodes = node_ids[:, None] != node_ids[None, :]
        both_valid = valid_nodes[:, None] & valid_nodes[None, :]
        pair_mask = same_graph & different_nodes & both_valid

        # Spherical harmonics Y_lm(r_hat_ij).
        irreps = e3j.Irreps.spherical_harmonics(self.l_max)
        harmonics = SphericalHarmonic(
            irreps,
            normalise=True,
            normalisation="integral",
        ).create_tensor(pair_vectors)

        # Degree l associated with each spherical-harmonic component.
        l_values = jnp.arange(self.l_max + 1)
        l_degrees = jnp.repeat(
            l_values,
            2 * l_values + 1,
            total_repeat_length=irreps.dim,
        )

        # Y_lm(r_hat_ij) * ||r_ij||^l
        pair_features = harmonics.array * pair_distances[..., None] ** l_degrees

        # Zero self-pairs, cross-graph pairs, and padded-node pairs
        pair_features = jnp.where(
            pair_mask[..., None],
            pair_features,
            0.0,
        )

        # Sum over source atoms i
        long_range_array = jnp.sum(pair_features, axis=1)

        if self.normalise:
            num_partners = jnp.sum(pair_mask, axis=1)
            num_partners = jnp.maximum(num_partners, 1)
            long_range_array = long_range_array / num_partners[:, None]

        long_range = e3j.IrrepsArray(
            irreps,
            long_range_array,
        )

        # Raw Fourier time embedding, one vector per node.
        # Shape: (n_node, time_embedding_dim)
        time_embedding = graph.nodes[self.time_field]

        # Every Fourier component is an invariant even scalar.
        time_irreps = e3j.IrrepsArray(
            f"{time_embedding.shape[-1]}x0e",
            time_embedding,
        )

        # Explicit time × AllPairs tensor product.
        long_range = e3j.tensor_product(
            time_irreps,
            long_range,
        )

        nodes = graph.nodes
        nodes[self.field] = e3j.concatenate([nodes[self.field], long_range])
        return graph._replace(nodes=nodes)
