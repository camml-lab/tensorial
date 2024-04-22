# -*- coding: utf-8 -*-
from typing import Optional, Union

import e3nn_jax as e3j
from flax import linen
import jax
import jax.numpy as jnp
import jraph

import tensorial

from . import keys

__all__ = 'NodewiseLinear', 'NodewiseReduce', 'NodewiseEncoding', 'NodewiseDecoding'


class NodewiseLinear(linen.Module):
    """Nodewise linear operation"""

    irreps_out: Union[str, e3j.Irreps]
    irreps_in: Optional[e3j.Irreps] = None
    field: str = keys.FEATURES
    out_field: Optional[str] = keys.FEATURES

    def setup(self):
        # pylint: disable=attribute-defined-outside-init
        self.linear = e3j.flax.Linear(
            irreps_out=self.irreps_out,
            irreps_in=self.irreps_in,
            force_irreps_out=True,
        )

    def __call__(self, graph: jraph.GraphsTuple):
        nodes = graph.nodes
        nodes[self.out_field] = self.linear(nodes[self.field])
        return graph._replace(nodes=nodes)


class NodewiseReduce(linen.Module):
    """Nodewise reduction operation. Saved to a global value"""

    field: str
    out_field: Optional[str] = None
    reduce: str = 'sum'
    average_num_atoms: float = None

    def setup(self):
        # pylint: disable=attribute-defined-outside-init
        if self.reduce not in ('sum', 'mean', 'normalized_sum'):
            raise ValueError(self.reduce)

        self._out_field = self.out_field or f'{self.reduce}_self.{self.field}'

        if self.reduce == 'normalized_sum':
            if self.average_num_atoms is None:
                raise ValueError(self.average_num_atoms)
            self.constant = float(self.average_num_atoms)**-0.5
            self._reduce = 'sum'
        else:
            self.constant = 1.0
            self._reduce = self.reduce

    def __call__(self, graph: jraph.GraphsTuple, key=None) -> jraph.GraphsTuple:
        inputs = graph.nodes[self.field] if self.field is not None else graph.nodes

        # this aggregation follows jraph/_src/models.py
        n_graph = graph.n_node.shape[0]
        graph_idx = jnp.arange(n_graph)
        sum_n_node = jax.tree_util.tree_leaves(graph.nodes)[0].shape[0]
        node_gr_idx = jnp.repeat(graph_idx, graph.n_node, axis=0, total_repeat_length=sum_n_node)

        reduced = jax.tree_util.tree_map(lambda n: jraph.segment_sum(n, node_gr_idx, n_graph), inputs)

        globals_dict = graph.globals or {}
        globals_dict[self._out_field] = reduced
        replacements = dict(globals=globals_dict)

        return graph._replace(**replacements)


class NodewiseEncoding(linen.Module):
    """
    Take the attributes in the nodes dictionary given by attrs, encode them, and store the results as a direct sum
    of irreps in the out_field.
    """

    attrs: tensorial.IrrepsTree
    out_field: str = keys.ATTRIBUTES

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        print(f'JAX compiling {self.__class__.__name__}')

        # Create the encoding
        encoded = tensorial.create_tensor(self.attrs, graph.nodes)
        # Store in output field
        nodes = graph.nodes
        nodes[self.out_field] = encoded
        return graph._replace(nodes=nodes)


class NodewiseDecoding(linen.Module):
    """
    Decode the direct sum of irreps stored in the in_field and store each tensor as a node value with key coming from
    the attrs.
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
            tensor_slice = irreps_tensor[..., idx:idx + irreps.dim]
            nodes_dict[key] = tensorial.from_tensor(value, tensor_slice)
            idx += irreps.dim

        # All done, return the new graph
        return graph._replace(nodes=nodes_dict)
