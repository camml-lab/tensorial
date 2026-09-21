"""Abstract interfaces for neighbour lists and neighbour finders.

These are the contracts implemented by the concrete backends in
:mod:`tensorial.geometry.np_neighbours` (NumPy-based) and
:mod:`tensorial.geometry.jax_neighbours` (JAX-based).
"""

import abc
import collections

import jax

__all__ = ("Edges", "NeighbourList", "NeighbourFinder")

Edges = collections.namedtuple("Edge", "from_idx to_idx cell_shift")


class NeighbourList(abc.ABC):
    """Abstract representation of a neighbour list for a set of particles.

    Concrete subclasses must implement :attr:`num_particles`,
    :attr:`max_neighbours` and :meth:`get_edges`.
    """

    @property
    @abc.abstractmethod
    def num_particles(self) -> int:
        """The number of particles for which neighbours are stored."""

    @property
    @abc.abstractmethod
    def max_neighbours(self) -> int:
        """The (observed) maximum number of neighbours of any single particle."""

    @abc.abstractmethod
    def get_edges(self) -> Edges:
        """Return all (particle, neighbour) pairs together with their periodic-image shifts."""


class NeighbourFinder(abc.ABC):
    """Abstract contract for finding a neighbour list around a set of positions."""

    @abc.abstractmethod
    def get_neighbours(
        self, positions: jax.typing.ArrayLike, max_neighbours: int = None
    ) -> NeighbourList:
        """Build the :class:`NeighbourList` containing all neighbours within the cutoff."""
