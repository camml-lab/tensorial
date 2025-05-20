from collections.abc import Mapping, Sequence
from typing import Optional, Union

import beartype
import jax
import jax.numpy as jnp
import jaxtyping as jt
import jraph
from pytray import tree
import reax
from typing_extensions import override

from . import keys
from .. import _common
from .. import keys as graph_keys
from .. import metrics
from ... import nn_utils, typing

__all__ = (
    "AllAtomicNumbers",
    "NumSpecies",
    "ForceStd",
    "AvgNumNeighbours",
    "AvgNumNeighboursByAtomType",
    "TypeContributionLstsq",
    "EnergyContributionLstsq",
)


def get(mapping: Mapping, key: str):
    try:
        return mapping[key]
    except KeyError:
        raise reax.exceptions.DataNotFound(f"Missing key: {key}") from None


AllAtomicNumbers = reax.metrics.Unique.from_fun(
    lambda graph, *_: (get(graph.nodes, keys.ATOMIC_NUMBERS), graph.nodes.get(graph_keys.MASK))
)

NumSpecies = reax.metrics.NumUnique.from_fun(
    lambda graph, *_: (get(graph.nodes, keys.ATOMIC_NUMBERS), graph.nodes.get(graph_keys.MASK))
)


ForceStd = reax.metrics.Std.from_fun(
    lambda graph, *_: (get(graph.nodes, keys.FORCES), graph.nodes.get(graph_keys.MASK))
)

AvgNumNeighbours = reax.metrics.Average.from_fun(
    lambda graph, *_: (
        jnp.bincount(graph.senders, length=jnp.sum(graph.n_node)),
        graph.nodes.get(graph_keys.MASK),
    )
)


class EnergyPerAtomLstsq(reax.metrics.FromFun):
    """Calculate the least squares estimate of the energy per atom"""

    metric = reax.metrics.LeastSquaresEstimate()

    @staticmethod
    def fun(graph, *_):
        return graph.n_node.reshape(-1, 1), graph.globals[keys.TOTAL_ENERGY].reshape(-1)

    def compute(self) -> jax.Array:
        return super().compute().reshape(())


class TypeContributionLstsq(reax.metrics.Metric[jax.typing.ArrayLike]):
    type_counts: Optional[typing.ArrayType] = None
    values: Optional[typing.ArrayType] = None
    mask: Optional[typing.ArrayType] = None

    @property
    def is_empty(self):
        return self.type_counts is None

    @jt.jaxtyped(typechecker=beartype.beartype)
    def create(
        # pylint: disable=arguments-differ
        self,
        type_counts: jt.Float[typing.ArrayType, "batch_size ..."],
        values: jt.Float[typing.ArrayType, "batch_size ..."],
        mask: jt.Bool[typing.ArrayType, "batch_size ..."] = None,
    ) -> "TypeContributionLstsq":
        return TypeContributionLstsq(type_counts, values, mask)

    @jt.jaxtyped(typechecker=beartype.beartype)
    def update(
        # pylint: disable=arguments-differ
        self,
        type_counts: jt.Float[typing.ArrayType, "batch_size ..."],
        values: jt.Float[typing.ArrayType, "batch_size ..."],
        mask: jt.Bool[typing.ArrayType, "batch_size ..."] = None,
    ) -> "TypeContributionLstsq":
        if self.is_empty:
            return self.create(type_counts, values)  # pylint: disable=not-callable

        return TypeContributionLstsq(
            type_counts=jnp.stack((self.type_counts, values)),
            values=jnp.stack((self.values, values)),
            mask=jnp.concatenate((self.mask, mask)),
        )

    def merge(self, other: "TypeContributionLstsq") -> "TypeContributionLstsq":
        if self.is_empty:
            return other
        if other.is_empty:
            return self

        return TypeContributionLstsq(
            type_counts=jnp.vstack((self.type_counts, other.type_counts)),
            values=jnp.vstack((self.values, other.values)),
            mask=jnp.concatenate((self.mask, other.mask)),
        )

    def compute(self):
        if self.is_empty:
            raise RuntimeError("This metric is empty, cannot compute!")

        # Check if we should mask off unused values
        if self.mask is None:
            type_counts = self.type_counts
            values = self.values
        else:
            type_counts = self.type_counts[self.mask]  # pylint: disable=unsubscriptable-object
            values = self.values[self.mask]  # pylint: disable=unsubscriptable-object

        return jnp.linalg.lstsq(type_counts, values)[0]


class EnergyContributionLstsq(reax.Metric):
    _type_map: jax.typing.ArrayLike
    _metric: Optional[TypeContributionLstsq] = None

    def __init__(self, type_map: Sequence, metric: TypeContributionLstsq = None):
        if type_map is None:
            raise ValueError("Must supply a value type_map")
        self._type_map = jnp.asarray(type_map)
        self._metric = metric

    def empty(self) -> "EnergyContributionLstsq":
        if self._metric is None:
            return self

        return EnergyContributionLstsq(self._type_map)

    def merge(self, other: "EnergyContributionLstsq") -> "EnergyContributionLstsq":
        if other._metric is None:  # pylint: disable=protected-access
            return self
        if self._metric is None:
            return other

        return EnergyContributionLstsq(
            type_map=self._type_map,
            metric=self._metric.merge(other._metric),  # pylint: disable=protected-access
        )

    @override
    def create(  # pylint: disable=arguments-differ
        self, graphs: jraph.GraphsTuple, *_
    ) -> "EnergyContributionLstsq":
        val = self._fun(graphs)  # pylint: disable=not-callable
        return type(self)(type_map=self._type_map, metric=TypeContributionLstsq(*val))

    @override
    def update(  # pylint: disable=arguments-differ
        self, graphs: jraph.GraphsTuple, *_
    ) -> "EnergyContributionLstsq":
        if self._metric is None:
            return self.create(graphs)

        val = self._fun(graphs)  # pylint: disable=not-callable
        return EnergyContributionLstsq(type_map=self._type_map, metric=self._metric.update(*val))

    @override
    def compute(self):
        if self._metric is None:
            raise RuntimeError("Nothing to compute, metric is empty!")

        return self._metric.compute()

    @jt.jaxtyped(typechecker=beartype.beartype)
    def _fun(self, graphs: jraph.GraphsTuple, *_) -> tuple[
        jt.Float[typing.ArrayType, "batch_size k"],
        jt.Float[typing.ArrayType, "batch_size 1"],
        Optional[jt.Bool[typing.ArrayType, "batch_size"]],
    ]:
        graph_dict = graphs._asdict()
        num_nodes = graphs.n_node

        try:
            types = tree.get_by_path(graph_dict, ("nodes", keys.ATOMIC_NUMBERS))
        except KeyError:
            raise reax.exceptions.DataNotFound(
                f"Missing key: {('nodes', keys.TOTAL_ENERGY)}"
            ) from None

        if self._type_map is None:
            num_classes = types.max().item() + 1  # Assume the types go 0,1,2...N
        else:
            # Transform the atomic numbers from whatever they are to 0, 1, 2....
            types = nn_utils.vwhere(types, self._type_map)
            num_classes = len(self._type_map)

        one_hots = jax.nn.one_hot(types, num_classes)

        # TODO: make it so we don't need to set the value in the graph
        one_hot_field = ("type_one_hot",)
        tree.set_by_path(graphs.nodes, one_hot_field, one_hots)
        type_counts = _common.reduce(graphs, ("nodes",) + one_hot_field, reduction="sum")

        # Predicting values
        try:
            values = tree.get_by_path(graph_dict, ("globals", keys.TOTAL_ENERGY))
        except KeyError:
            raise reax.exceptions.DataNotFound(
                f"Missing key: {('globals', keys.TOTAL_ENERGY)}"
            ) from None

        if graph_keys.MASK in graph_dict["globals"]:
            mask = graph_dict["globals"][graph_keys.MASK]
        else:
            mask = None

        # Normalise by number of nodes
        type_counts = jax.vmap(lambda numer, denom: numer / denom, (0, 0))(type_counts, num_nodes)
        values = jax.vmap(lambda numer, denom: numer / denom, (0, 0))(values, num_nodes)

        return type_counts, values, mask


class AvgNumNeighboursByAtomType(metrics.AvgNumNeighboursByType):
    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        atom_types: Union[Sequence[int], jt.Int[jt.Array, "n_types"]],
        type_field: str = keys.ATOMIC_NUMBERS,
        state: Optional[metrics.AvgNumNeighboursByType.Averages] = None,
    ):
        super().__init__(atom_types, type_field, state)


reax.metrics.get_registry().register_many(
    {
        "atomic/num_species": NumSpecies,
        "atomic/all_atomic_numbers": AllAtomicNumbers,
        "atomic/avg_num_neighbours": AvgNumNeighbours,
        "atomic/force_std": ForceStd,
        "atomic/energy_per_atom_lstsq": EnergyPerAtomLstsq,
    }
)
