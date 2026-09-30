from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import beartype
import jax
import jax.numpy as jnp
import jaxtyping as jt
from jaxtyping import Int
import jraph
import reax
from typing_extensions import override

from tensorial.typing import Array

from .. import keys as _keys
from .. import keys as graph_keys
from .. import metrics
from . import keys

if TYPE_CHECKING:
    from tensorial import gcnn

__all__ = (
    "AllAtomicNumbers",
    "NumSpecies",
    "ForceStd",
    "AvgNumNeighbours",
    "AvgNumNeighboursByAtomType",
    "EnergyContributionLstsq",
    "EnergyPerAtomLstsq",
)


def get(mapping: Mapping, key: str):
    try:
        return mapping[key]
    except KeyError:
        raise reax.exceptions.DataNotFound(f"Missing key: {key}") from None


AllAtomicNumbers = reax.metrics.Unique.from_fun(
    lambda graph, *_: (get(graph.nodes, keys.ATOMIC_NUMBERS), graph.nodes.get(graph_keys.MASK)),
    name="AtomicNumbers",
)


NumSpecies = reax.metrics.NumUnique.from_fun(
    lambda graph: (get(graph.nodes, keys.ATOMIC_NUMBERS), graph.nodes.get(graph_keys.MASK)),
    name="Species",
)


ForceStd = reax.metrics.Std.from_fun(
    lambda graph: (get(graph.nodes, keys.FORCES), graph.nodes.get(graph_keys.MASK)), name="Force"
)


AvgNumNeighbours = reax.metrics.Average.from_fun(
    lambda graph, *_: (
        jnp.bincount(graph.senders, length=graph.nodes[_keys.POSITIONS].shape[0]),
        graph.nodes.get(graph_keys.MASK),
    )
)


class EnergyPerAtomLstsq(reax.metrics.FromFun):
    """Calculate the least squares estimate of the energy per atom"""

    metric = reax.metrics.LeastSquaresEstimate

    @staticmethod
    def func(graph, *_):
        return graph.n_node.reshape(-1, 1), graph.globals[keys.TOTAL_ENERGY].reshape(-1)

    def compute(self) -> jax.Array:
        return super().compute().reshape(())


class EnergyContributionLstsq(metrics.PropertyContributionLstsq):
    def __init__(
        self, type_map: Sequence | Array, metric: "gcnn.metrics.TypeContributionLstsq | None" = None
    ):
        super().__init__(
            type_key=keys.ATOMIC_NUMBERS,
            property_key=keys.TOTAL_ENERGY,
            type_source="nodes",
            type_map=type_map,
            normalize=True,
            metric=metric,
        )

    @override
    def create(self, graphs: jraph.GraphsTuple, /, *_, **__) -> "EnergyContributionLstsq":
        val = self._fun(graphs)
        return EnergyContributionLstsq(
            type_map=self._type_map, metric=metrics.TypeContributionLstsq.create(*val)
        )

    @override
    def update(self, graphs: jraph.GraphsTuple, /, *_, **__) -> "EnergyContributionLstsq":
        if self._metric is None:
            return self.create(graphs)
        val = self._fun(graphs)
        return EnergyContributionLstsq(type_map=self._type_map, metric=self._metric.update(*val))


class AvgNumNeighboursByAtomType(metrics.AvgNumNeighboursByType):
    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        atom_types: Sequence[int] | Int[Array, "n_types"],
        type_field: str = keys.ATOMIC_NUMBERS,
        state: metrics.AvgNumNeighboursByType.Averages | None = None,
    ):
        super().__init__(atom_types, type_field, state)
