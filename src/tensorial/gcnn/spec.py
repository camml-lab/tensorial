from typing import TYPE_CHECKING, Sequence

import e3nn_jax as e3j
import jax
import jax.numpy as jnp
import jraph

from .. import base, tensors

if TYPE_CHECKING:
    import tensorial


class GraphSpec:
    """Specification for the data types carried on a graph"""

    def __init__(
        self,
        nodes: "tensorial.IrrepsObj" = None,
        edges: "tensorial.Tensorial" = None,
        globals: "tensorial.Tensorial" = None,
    ):  # pylint: disable=redefined-builtin
        self._nodes = nodes
        self._edges = edges
        self._globals = globals

    @property
    def nodes(self) -> "tensorial.Tensorial":
        return self._nodes

    @property
    def edges(self) -> "tensorial.Tensorial":
        return self._edges

    @property
    def globals(self) -> "tensorial.Tensorial":
        return self._globals

    def from_jraph(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        replacements = {}
        if self.nodes is not None:
            replacements["nodes"] = base.create(self.nodes, graph.nodes)
        if self.edges is not None:
            replacements["edges"] = base.create(self.edges, graph.edges)
        if self.globals is not None:
            replacements["globals"] = base.create(self.globals, graph.globals)

        return graph._replace(**replacements)


class SpeciesOneHot(tensors.OneHot):
    """One-hot encoding of species as a direct sum of scalars"""

    def __init__(self, species: Sequence):
        """The species supported by this one-hot encoding"""
        self._species = tuple(species)
        super().__init__(len(self._species))

    @property
    def species(self) -> tuple:
        return self._species

    def create_tensor(self, value) -> jnp.array:
        # return self.species.index(value)
        return e3j.IrrepsArray(self.irreps, jax.nn.one_hot(value, self.num_classes))
