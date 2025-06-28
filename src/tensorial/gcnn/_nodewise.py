import logging
from typing import TYPE_CHECKING, Optional, Union

import e3nn_jax as e3j
from flax import linen
import jax.numpy as jnp
import jraph

from . import _base, _common, _experimental, _tree, keys, utils
from .. import base

if TYPE_CHECKING:
    import tensorial

__all__ = (
    "NodewiseLinear",
    "NodewiseReduce",
    "NodewiseEmbedding",
    "NodewiseEncoding",
    "NodewiseDecoding",
)

_LOGGER = logging.getLogger(__name__)


class NodewiseLinear(linen.Module):
    """Nodewise linear operation"""

    irreps_out: Union[str, e3j.Irreps]
    irreps_in: Optional[e3j.Irreps] = None
    field: str = keys.FEATURES
    out_field: Optional[str] = keys.FEATURES
    num_types: Optional[int] = None
    types_field: Optional[str] = None

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
                    "Ignoring."
                )
            self._types_field = self.types_field

    @_base.shape_check
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        nodes = graph.nodes
        if self.num_types:
            # We are using weights indexed by the type
            features = self.linear(nodes[self._types_field], nodes[self.field])
        else:
            features = self.linear(nodes[self.field])

        nodes[self.out_field] = features
        return graph._replace(nodes=nodes)


class NodewiseReduce(linen.Module):
    """
    Applies a reduction operation over node features and stores the result in the graph globals.

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
    out_field: Optional[str] = None
    reduce: str = "sum"
    average_num_atoms: float = None

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
        reduced = self.constant * _common.reduce(graph, self._field, self._reduce)
        return _experimental.update_graph(graph).set(self._out_field, reduced).get()


class NodewiseEmbedding(linen.Module):
    """
    Take the attributes in the nodes dictionary given by attrs, embed them, and store the results
    as a direct sum of irreps in the out_field.
    """

    attrs: "tensorial.IrrepsTree"
    out_field: str = keys.ATTRIBUTES
    node_shape_from: Optional[str] = keys.POSITIONS

    @_base.shape_check
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        if isinstance(self.attrs, (dict, linen.FrozenDict)):
            values = {}
            for key in self.attrs.keys():
                path = _tree.path_from_str(key)
                if len(path) > 1:
                    if not path[0] in ("nodes", "globals"):
                        raise ValueError(f"The attribute key must not contain '.', got: {key}")
                else:
                    # Assume it is a node attribute
                    path = ("nodes",) + path

                value = _tree.get(graph, path)
                if path[0] == "globals":
                    if self.node_shape_from is None:
                        # This will not have a static shape, so will cause recompilation
                        total_repeat_length = jnp.sum(graph.n_node)
                    else:
                        total_repeat_length = graph.nodes[self.node_shape_from].shape[0]

                    value = jnp.repeat(
                        value, graph.n_node, axis=0, total_repeat_length=total_repeat_length
                    )

                values[key] = value
        else:
            values = graph.nodes

        # Create the embedding
        encoded = base.create_tensor(self.attrs, values)
        # Store in output field
        nodes = graph.nodes
        nodes[self.out_field] = encoded
        return graph._replace(nodes=nodes)


class NodewiseDecoding(linen.Module):
    """
    Decode the direct sum of irreps stored in the in_field and store each tensor as a node value
    with key coming from the attrs.
    """

    attrs: "tensorial.IrrepsTree"
    in_field: str = keys.ATTRIBUTES

    @_base.shape_check
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        # Here, we need to split up the direct sum of irreps in the in field, and save the values
        # in the nodes dict corresponding to the attrs keys
        idx = 0
        nodes_dict = graph.nodes
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
