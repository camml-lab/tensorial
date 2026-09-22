from flax import linen
import jraph
from typing_extensions import override

from . import keys
from .. import gcnn
from .keys import predicted

__all__ = ("CalcForces",)


class CalcForces(linen.Module):
    energy_fn: gcnn.GraphFunction
    out_key: gcnn.TreePathLike | None = predicted(keys.FORCES)

    @override
    def setup(self):
        """Initializes the internal virial function used to compute stress-related quantities."""
        # pylint: disable=attribute-defined-outside-init

        self._forces_fn = gcnn.diff(
            self.energy_fn,
            f"globals.{keys.predicted(keys.TOTAL_ENERGY)}:gk",
            wrt="nodes.positions:Ia",
            out=":Ia",
            scale=-1.0,
            return_graph=True,
        )

    @override
    def __call__(  # pylint: disable=arguments-differ
        self, graph: jraph.GraphsTuple, /
    ) -> jraph.GraphsTuple:
        forces, graph = self._forces_fn(graph, graph.nodes[keys.POSITIONS])
        return gcnn.experimental.update_graph(graph).set(("nodes", self.out_key), forces).get()
