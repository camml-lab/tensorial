from collections.abc import Callable
import functools
from typing import TYPE_CHECKING

import e3nn_jax as e3j
from flax import linen
import jax
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

    irreps_out: str | e3j.Irreps
    irreps_in: e3j.Irreps | None = None
    field: str = keys.FEATURES
    out_field: str | None = keys.FEATURES

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
    """Embed edge attributes into a direct sum of irreps.

    The edge attributes specified by ``attrs`` (a mapping of attribute name to a
    :class:`~tensorial.base.Attr` or irreps specification) are encoded and stored in a
    single ``e3nn_jax.IrrepsArray`` at ``out_field`` in the graph's edges.

    Args:
        attrs: a mapping of edge attribute name to the tensorial attribute (or irreps)
            used to encode it
        out_field: the edge field in which to store the encoded features

    Example:
        >>> embedding = EdgewiseEmbedding({"vectors": "1o"})
    """

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
    """Decode the direct sum of irreps stored in the in_field and store each tensor as a node value
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
    """Embed edge lengths into a radial basis.

    Computes the edge vectors (and hence lengths) of the graph, then expands the lengths
    in a Bessel radial basis of ``num_basis`` functions, optionally multiplied by a
    polynomial envelope that smoothly approaches zero at the cutoff. The result is stored
    at ``out_field`` in the graph's edges.

    Args:
        field: the edge field from which to read the lengths (used if already present)
        out_field: the edge field in which to store the radial embeddings
        num_basis: the number of radial basis functions
        r_max: the cutoff radius, used to scale the Bessel basis and envelope
        envelope: if ``True``, use a default polynomial envelope; or a callable
            ``envelope(r / r_max)`` to use instead

    Example:
        >>> embedding = RadialBasisEdgeEmbedding(num_basis=8, r_max=5.0, envelope=True)
    """

    field: str = keys.EDGE_LENGTHS
    out_field: str = keys.RADIAL_EMBEDDINGS
    num_basis: int = 8
    r_max: float = 4.0
    envelope: bool | Callable = False

    def setup(self):
        # pylint: disable=attribute-defined-outside-init
        self.radial_embedding = functools.partial(  # pylint: disable=attribute-defined-outside-init
            e3j.bessel,
            x_max=self.r_max,
            n=self.num_basis,
        )
        self._envelope = self._init_envelope(self.envelope)

    @staticmethod
    def _init_envelope(envelope) -> Callable | None:
        if envelope:
            return envelope if callable(envelope) else e3j.poly_envelope(1, 1)

        return None

    @_base.shape_check
    def __call__(
        self, graph: jraph.GraphsTuple
    ) -> jraph.GraphsTuple:  # pylint: disable=arguments-differ
        edge_dict = _spatial.with_edge_vectors(graph).edges
        r = base.as_array(edge_dict[keys.EDGE_LENGTHS])[:, 0]
        embedded = self.radial_embedding(r)
        if self._envelope is not None:
            embedded = jax.vmap(self._envelope)(r / self.r_max)[..., None] * embedded

        edge_dict[self.out_field] = embedded
        return graph._replace(edges=edge_dict)


class EdgeVectors(linen.Module):
    """Create edge vectors from atomic positions.  This will take into account the unit cell
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
