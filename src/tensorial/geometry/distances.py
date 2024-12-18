import abc
import collections

import jax

__all__ = ("Edges", "NeighbourList", "NeighbourFinder")

Edges = collections.namedtuple("Edge", "from_idx to_idx cell_shift")


class NeighbourList(abc.ABC):
    """An interface that represents a neighbour list"""

    @property
    @abc.abstractmethod
    def num_particles(self) -> int:
        """Get the number of neighbours in this list"""

    @property
    @abc.abstractmethod
    def max_neighbours(self) -> int:
        """Get the maximum number of neighbours in this list"""

    @abc.abstractmethod
    def get_edges(self) -> Edges:
        """Get the edges representing all neighbours"""


class NeighbourFinder(abc.ABC):
    @abc.abstractmethod
    def get_neighbours(
        self, positions: jax.typing.ArrayLike, max_neighbours: int = None
    ) -> NeighbourList:
        """Get the neighbour list for the given positions"""
