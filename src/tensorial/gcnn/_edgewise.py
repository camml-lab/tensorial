# -*- coding: utf-8 -*-
import functools

import e3nn_jax as e3j
from flax import linen
import jraph

import tensorial

from . import _graphs, keys

__all__ = 'EdgewiseEncoding', 'RadialBasisEdgeEncoding', 'EdgeVectors'


class EdgewiseEncoding(linen.Module):
    attrs: tensorial.IrrepsTree
    out_field: str = keys.ATTRIBUTES

    def __call__(self, graph: jraph.GraphsTuple, key=None) -> jraph.GraphsTuple:  # pylint: disable=arguments-differ
        # Create the encoding
        encoded = tensorial.create_tensor(self.attrs, graph.edges)
        # Store in output field
        edges = graph.edges
        edges[self.out_field] = encoded
        return graph._replace(edges=edges)


class RadialBasisEdgeEncoding(linen.Module):
    field: str = keys.EDGE_LENGTHS
    out_field: str = keys.RADIAL_EMBEDDINGS
    num_basis: int = 8
    r_max: float = 4.0

    def setup(self):
        self.radial_embedding = functools.partial(  # pylint: disable=attribute-defined-outside-init
            e3j.bessel,
            x_max=self.r_max,
            n=self.num_basis,
        )

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:  # pylint: disable=arguments-differ
        edge_dict = _graphs.with_edge_vectors(graph).edges
        edge_dict[self.out_field] = self.radial_embedding(edge_dict[keys.EDGE_LENGTHS][:, 0])
        return graph._replace(edges=edge_dict)


class EdgeVectors(linen.Module):
    """Create edge vectors from atomic positions.  This will take into account the unit cell (if present)"""

    @linen.compact
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:  # pylint: disable=arguments-differ
        return _graphs.with_edge_vectors(graph, with_lengths=True)
