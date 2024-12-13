import abc

import jax


class NeighbourFinder(abc.ABC):
    def get_neighbours(
        self, positions: jax.typing.ArrayLike, max_neighbours: int = None
    ) -> NeighbourList:
        """Get the neighbour list for the given positions"""

    def estimate_neighbours(self, positions: jax.typing.ArrayLike) -> int:
        """Estimate the number of neighbours per particle"""
