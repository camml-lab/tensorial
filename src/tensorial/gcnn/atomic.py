from collections.abc import Iterable
import numbers
from typing import Any, Hashable, Mapping, MutableMapping, Optional, Sequence, Union

import beartype
import equinox
import jax
import jax.numpy as jnp
import jaxtyping as jt
import jraph
import numpy as np
from pytray import tree
import reax

from tensorial import base, nn_utils, typing

from . import _common, _graphs, _modules, _typing, keys, metrics, utils

ENERGY_PER_ATOM = "energy/atom"
TOTAL_ENERGY = "energy"
FORCES = "forces"
STRESS = "stress"
VIRIAL = "virial"
PBC = "pbc"
ATOMIC_NUMBERS = "atomic_numbers"

# Global quantities
ASE_GLOBAL_KEYS = {"energy", "free_energy", "stress", "magmom"}
# Per-atom quantities
ASE_ATOM_KEYS = {"numbers", "forces", "stresses", "charges", "magmoms", "energies"}

PyTree = Any


@jt.jaxtyped(typechecker=beartype.beartype)
def graph_from_ase(
    ase_atoms: "ase.atoms.Atoms",
    r_max: numbers.Number,
    *,
    key_mapping: Optional[dict[str, str]] = None,
    atom_include_keys: Optional[Iterable] = ("numbers",),
    edge_include_keys: Optional[Iterable] = tuple(),
    global_include_keys: Optional[Iterable] = tuple(),
    cell: Optional[typing.CellType] = None,
    pbc: Optional[Union[bool, typing.PbcType]] = None,
    **kwargs,
) -> jraph.GraphsTuple:
    """
    Create a jraph Graph from an ase.Atoms object

    :param ase_atoms: the Atoms object
    :param r_max: the maximum neighbour distance to use when considering two atoms to be neighbours
    :param key_mapping:
    :param atom_include_keys:
    :param global_include_keys:
    :param cell: an optional unit cell (otherwise will be taken from ase.cell)
    :param pbc: an optional periodic boundary conditions array [bool, bool, bool] (otherwise will be
        taken from ase.pbc)
    :return: the atomic graph
    """
    # pylint: disable=too-many-branches
    from ase.calculators import singlepoint
    import ase.stress

    key_mapping = key_mapping or {}
    _key_mapping = {
        "forces": FORCES,
        "energy": TOTAL_ENERGY,
        "numbers": ATOMIC_NUMBERS,
    }
    _key_mapping.update(key_mapping)
    key_mapping = _key_mapping
    del _key_mapping

    graph_globals = {}
    for key in global_include_keys:
        get_attrs(graph_globals, ase_atoms.arrays, key, key_mapping)

    atoms = {}
    for key in atom_include_keys:
        get_attrs(atoms, ase_atoms.arrays, key, key_mapping)

    edges = {}
    for key in edge_include_keys:
        get_attrs(edges, ase_atoms.arrays, key, key_mapping)

    if ase_atoms.calc is not None:
        if not isinstance(
            ase_atoms.calc,
            (singlepoint.SinglePointCalculator, singlepoint.SinglePointDFTCalculator),
        ):
            raise NotImplementedError(
                f"`from_ase` does not support calculator {type(ase_atoms.calc).__name__}"
            )

        for key, val in ase_atoms.calc.results.items():
            if key in atom_include_keys:
                atoms[key] = base.atleast_1d(val, np_=np)
            elif key in global_include_keys:
                graph_globals[key] = base.atleast_1d(val, np_=np)

    # Transform ASE-style 6 element Voigt order stress to Cartesian
    for key in (STRESS, VIRIAL):
        if key in graph_globals:
            if graph_globals[key].shape == (3, 3):
                # In the format we want
                pass
            elif graph_globals[key].shape == (6,):
                # In Voigt order
                graph_globals[key] = ase.stress.voigt_6_to_full_3x3_stress(graph_globals[key])
            else:
                raise RuntimeError(f"Unexpected shape for {key}, got: {graph_globals[key].shape}")

    # cell and pbc in kwargs can override the ones stored in atoms
    cell = cell or ase_atoms.get_cell()
    pbc = pbc or ase_atoms.pbc

    return _graphs.graph_from_points(
        pos=ase_atoms.positions,
        fractional_positions=False,
        r_max=r_max,
        cell=cell.__array__(),
        pbc=pbc,
        nodes=atoms,
        edges=edges,
        graph_globals=graph_globals,
        **kwargs,
    )


def get_attrs(store_in: MutableMapping, get_from: Mapping, key: Hashable, key_map: Mapping) -> bool:
    out_key = key_map.get(key, key)
    try:
        value = get_from[key]
    except KeyError:
        # Couldn't find the attribute
        return False

    store_in[out_key] = value
    return True


@jt.jaxtyped(typechecker=beartype.beartype)
class SpeciesTransform(equinox.Module):
    """
    Take an ordered list of species and transform them into an integer corresponding to their
    position in the list
    """

    atomic_numbers: jt.Int[jax.Array, "numbers"]
    field: str = ATOMIC_NUMBERS
    out_field: str = keys.SPECIES

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        atomic_numbers: Union[Sequence[int], jt.Int[typing.ArrayType, "numbers"]],
        field: str = ATOMIC_NUMBERS,
        out_field: str = keys.SPECIES,
    ):
        self.atomic_numbers = jnp.asarray(
            atomic_numbers
        )  # pylint: disable=attribute-defined-outside-init
        self.field = field
        self.out_field = out_field

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __call__(
        self, graph: jraph.GraphsTuple
    ) -> jraph.GraphsTuple:  # pylint: disable=arguments-differ
        nodes = graph.nodes
        nodes[self.out_field] = nn_utils.vwhere(nodes[self.field], self.atomic_numbers)

        return graph._replace(nodes=nodes)


def per_species_rescale(
    num_types: int,
    field: str,
    *,
    types_field: str = None,
    out_field: str = None,
    shifts: jax.typing.ArrayLike = None,
    scales: jax.typing.ArrayLike = None,
) -> _modules.IndexedRescale:
    types_field = types_field or ("nodes", keys.SPECIES)
    return _modules.IndexedRescale(
        num_types,
        index_field=types_field,
        field=field,
        out_field=out_field,
        shifts=shifts,
        scales=scales,
    )


def estimate_species_contribution(
    graphs: jraph.GraphsTuple,
    value_field: _typing.TreePathLike,
    type_field: _typing.TreePathLike = ("nodes", keys.SPECIES),
    type_map: Sequence[int] = None,
) -> tuple[jax.Array, jax.Array]:
    """Estimates the contribution of the one hot encoded field to the final value

    :param graphs: a graphs tuple containing all the atomic structures
    :param value_field: the field containing the final values
    :param type_field: the field containing the type index of atomic species
    :return: the least squares contribution
    """
    graph_dict = graphs._asdict()
    value_field = utils.path_from_str(value_field)
    type_field = utils.path_from_str(type_field)
    num_nodes = graphs.n_node

    type_values = tree.get_by_path(graph_dict, type_field)
    if type_map is not None:
        # Transform the atomic numbers into from whatever they are to 0, 1, 2....
        nn_utils.vwhere(type_values, type_map)

    num_classes = type_values.max().item() + 1  # Assume the types go 0,1,2...N
    one_hots = jax.nn.one_hot(type_values, num_classes)

    one_hot_field = ("type_one_hot",)
    tree.set_by_path(graphs.nodes, one_hot_field, one_hots)
    type_values = _common.reduce(graphs, ("nodes",) + one_hot_field, reduction="sum")

    # Predicting values
    values = tree.get_by_path(graph_dict, value_field)

    # Normalise by number of nodes
    type_values = jax.vmap(lambda numer, denom: numer / denom, (0, 0))(type_values, num_nodes)
    values = jax.vmap(lambda numer, denom: numer / denom, (0, 0))(values, num_nodes)

    contributions = jnp.linalg.lstsq(type_values, values)[0]
    estimates = type_values @ contributions
    stds = jnp.std(values - estimates)

    return contributions, stds


def energy_per_atom_lstsq(graphs: jraph.GraphsTuple, stats: dict[str, jnp.ndarray]) -> jax.Array:
    return estimate_species_contribution(
        graphs,
        value_field=("globals", "energy"),
        type_field=("nodes", ATOMIC_NUMBERS),
        type_map=stats["all_atomic_numbers"],
    )[0]


AllAtomicNumbers = reax.metrics.Unique.from_fun(
    lambda graph, *_: (graph.nodes[ATOMIC_NUMBERS], graph.nodes.get(keys.MASK))
)

NumSpecies = reax.metrics.NumUnique.from_fun(
    lambda graph, *_: (graph.nodes[ATOMIC_NUMBERS], graph.nodes.get(keys.MASK))
)

ForceStd = reax.metrics.Std.from_fun(
    lambda graph, *_: (graph.nodes[FORCES], graph.nodes.get(keys.MASK))
)

AvgNumNeighbours = reax.metrics.Average.from_fun(
    lambda graph, *_: (
        jnp.bincount(graph.senders, length=jnp.sum(graph.n_node)),
        graph.nodes.get(keys.MASK),
    )
)


class EnergyPerAtomLstsq(reax.metrics.FromFun):
    """Calculate the least squares estimate of the energy per atom"""

    metric = reax.metrics.LeastSquaresEstimate()

    @staticmethod
    def fun(graph, *_):
        return graph.n_node.reshape(-1, 1), graph.globals[TOTAL_ENERGY].reshape(-1)

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

    def create(  # pylint: disable=arguments-differ
        self, graphs: jraph.GraphsTuple, *_
    ) -> "EnergyContributionLstsq":
        val = self._fun(graphs)  # pylint: disable=not-callable
        return type(self)(type_map=self._type_map, metric=TypeContributionLstsq(*val))

    def update(  # pylint: disable=arguments-differ
        self, graphs: jraph.GraphsTuple, *_
    ) -> "EnergyContributionLstsq":
        if self._metric is None:
            return self.create(graphs)

        val = self._fun(graphs)  # pylint: disable=not-callable
        return EnergyContributionLstsq(type_map=self._type_map, metric=self._metric.update(*val))

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

        types = tree.get_by_path(graph_dict, ("nodes", ATOMIC_NUMBERS))
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
        values = tree.get_by_path(graph_dict, ("globals", TOTAL_ENERGY))
        if keys.MASK in graph_dict["globals"]:
            mask = graph_dict["globals"][keys.MASK]
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
        type_field: str = ATOMIC_NUMBERS,
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
