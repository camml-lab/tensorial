import functools
from typing import TYPE_CHECKING, Optional, Union

import e3nn_jax as e3j
from flax import linen
import jraph

from . import _base, _spatial, keys
from .. import base

if TYPE_CHECKING:
    import tensorial

__all__ = (
    "EdgewiseLinear",
    "EdgewiseDecoding",
    "EdgewiseEmbedding",
    "EdgewiseEncoding",
    "RadialBasisEdgeEmbedding",
    "RadialBasisEdgeEncoding",
    "EdgeVectors",
)


class EdgewiseLinear(linen.Module):
    """Edgewise linear operation"""

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

    @_base.shape_check
    def __call__(self, graph: jraph.GraphsTuple):
        edges = graph.edges
        edges[self.out_field] = self.linear(edges[self.field])
        return graph._replace(edges=edges)


class EdgewiseEmbedding(linen.Module):
    attrs: "tensorial.IrrepsTree"
    out_field: str = keys.ATTRIBUTES

    @_base.shape_check
    def __call__(
        self, graph: jraph.GraphsTuple
    ) -> jraph.GraphsTuple:  # pylint: disable=arguments-differ
        # Create the encoding
        encoded = base.create_tensor(self.attrs, graph.edges)
        # Store in output field
        edges = graph.edges
        edges[self.out_field] = encoded
        return graph._replace(edges=edges)


class EdgewiseDecoding(linen.Module):
    """
    Decode the direct sum of irreps stored in the in_field and store each tensor as a node value
    with key coming from the attrs.
    """

    attrs: "tensorial.IrrepsTree"
    in_field: str = keys.ATTRIBUTES

    @_base.shape_check
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        # Here, we need to split up the direct sum of irreps in the in field, and save the values
        # in the edges dict corresponding to the attrs keys
        idx = 0
        edges_dict = graph.edges
        irreps_tensor = edges_dict[self.in_field]
        for key, value in base.tensorial_attrs(self.attrs).items():
            irreps = base.irreps(value)
            tensor_slice = irreps_tensor[..., idx : idx + irreps.dim]
            edges_dict[key] = base.from_tensor(value, tensor_slice)
            idx += irreps.dim

        # All done, return the new graph
        return graph._replace(edges=edges_dict)


class RadialBasisEdgeEmbedding(linen.Module):
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

    @_base.shape_check
    def __call__(
        self, graph: jraph.GraphsTuple
    ) -> jraph.GraphsTuple:  # pylint: disable=arguments-differ
        edge_dict = _spatial.with_edge_vectors(graph).edges
        edge_dict[self.out_field] = self.radial_embedding(
            base.as_array(edge_dict[keys.EDGE_LENGTHS])[:, 0]
        )
        return graph._replace(edges=edge_dict)


class EdgeVectors(linen.Module):
    """
    Create edge vectors from atomic positions.  This will take into account the unit cell
    (if present)
    """

    as_irreps_arrays: bool = False

    @linen.compact
    @_base.shape_check
    def __call__(
        self, graph: jraph.GraphsTuple
    ) -> jraph.GraphsTuple:  # pylint: disable=arguments-differ
        return _spatial.with_edge_vectors(
            graph, with_lengths=True, as_irreps_array=self.as_irreps_arrays
        )


# For legacy reasons
EdgewiseEncoding = EdgewiseEmbedding
RadialBasisEdgeEncoding = RadialBasisEdgeEmbedding
