# -*- coding: utf-8 -*-
import logging
from typing import Optional, Union

import e3nn_jax as e3j
from flax import linen
import jraph
from pytray import tree

import tensorial

from . import _common, keys, utils

__all__ = "NodewiseLinear", "NodewiseReduce", "NodewiseEncoding", "NodewiseDecoding"

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

    def __call__(self, graph: jraph.GraphsTuple):
        nodes = graph.nodes
        if self.num_types:
            # We are using weights indexed by the type
            features = self.linear(nodes[self._types_field], nodes[self.field])
        else:
            features = self.linear(nodes[self.field])

        nodes[self.out_field] = features
        return graph._replace(nodes=nodes)


class NodewiseReduce(linen.Module):
    """Nodewise reduction operation. Saved to a global value"""

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
            self.out_field or f"{self.reduce}_self.{self.field}"
        )

        if self.reduce == "normalized_sum":
            if self.average_num_atoms is None:
                raise ValueError(self.average_num_atoms)
            self.constant = float(self.average_num_atoms) ** -0.5
            self._reduce = "sum"
        else:
            self.constant = 1.0
            self._reduce = self.reduce

    def __call__(self, graph: jraph.GraphsTuple, key=None) -> jraph.GraphsTuple:
        reduced = self.constant * _common.reduce(graph, self._field, self._reduce)
        updates = utils.UpdateDict(graph._asdict())
        if updates["globals"] is None:
            updates["globals"] = dict()
        tree.set_by_path(updates, self._out_field, reduced)
        return jraph.GraphsTuple(**updates._asdict())


class NodewiseEncoding(linen.Module):
    """
    Take the attributes in the nodes dictionary given by attrs, encode them, and store the results
    as a direct sum of irreps in the out_field.
    """

    attrs: tensorial.IrrepsTree
    out_field: str = keys.ATTRIBUTES

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        print(f"JAX compiling {self.__class__.__name__}")

        # Create the encoding
        encoded = tensorial.create_tensor(self.attrs, graph.nodes)
        # Store in output field
        nodes = graph.nodes
        nodes[self.out_field] = encoded
        return graph._replace(nodes=nodes)


class NodewiseDecoding(linen.Module):
    """
    Decode the direct sum of irreps stored in the in_field and store each tensor as a node value
    with key coming from the attrs.
    """

    attrs: tensorial.IrrepsTree
    in_field: str = keys.ATTRIBUTES

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        # Here, we need to split up the direct sum of irreps in the in field, and save the values
        # in the nodes dict corresponding to the attrs keys
        idx = 0
        nodes_dict = graph.nodes
        irreps_tensor = nodes_dict[self.in_field]
        for key, value in tensorial.tensorial_attrs(self.attrs).items():
            irreps = tensorial.irreps(value)
            tensor_slice = irreps_tensor[..., idx : idx + irreps.dim]
            nodes_dict[key] = tensorial.from_tensor(value, tensor_slice)
            idx += irreps.dim

        # All done, return the new graph
        return graph._replace(nodes=nodes_dict)
