from collections.abc import Iterable
import numbers
from typing import Any, Hashable, Mapping, MutableMapping, Optional, Sequence, Union

import beartype
import equinox
import jax
import jax.numpy as jnp
import jaxtyping as jt
import jraph
from pytray import tree

from tensorial import base, metrics, typing

from . import _common, _graphs, _modules, _typing, keys, utils

ENERGY_PER_ATOM = "energy/atom"
TOTAL_ENERGY = "total_energy"
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
                atoms[key] = base.atleast_1d(val)
            elif key in global_include_keys:
                graph_globals[key] = base.atleast_1d(val)

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
        vwhere = jax.vmap(lambda num: jnp.argwhere(num == self.atomic_numbers, size=1)[0])
        nodes = graph.nodes
        nodes[self.out_field] = vwhere(nodes[self.field])[:, 0]
        return graph._replace(nodes=nodes)


def per_species_rescale(
    num_types: int,
    field: str,
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
        vwhere = jax.vmap(lambda num: jnp.argwhere(num == type_map, size=1)[0])
        type_values = vwhere(type_values)[:, 0]

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


AllAtomicNumbers = metrics.from_fun(
    lambda graph: metrics.Unique.create(
        graph.nodes[ATOMIC_NUMBERS], mask=graph.nodes.get(keys.MASK)
    )
)

NumSpecies = metrics.from_fun(
    lambda graph: metrics.NumUnique.create(
        graph.nodes[ATOMIC_NUMBERS], mask=graph.nodes.get(keys.MASK)
    )
)

ForceStd = metrics.from_fun(
    lambda graph: metrics.Std.create(graph.nodes[FORCES], mask=graph.nodes.get(keys.MASK))
)

AvgNumNeighbours = metrics.from_fun(
    lambda graph: metrics.Average.create(
        jnp.unique(graph.senders, return_counts=True)[1], mask=graph.nodes.get(keys.MASK)
    )
)

EnergyPerAtomLstsq = metrics.from_fun(
    lambda graph: metrics.LeastSquaresEstimate.create(
        graph.nodes[keys.SPECIES],
        graph.globals[TOTAL_ENERGY],
    )
)

metrics.get_registry().register_many(
    {
        "atomic/num_species": NumSpecies,
        "atomic/all_atomic_numbers": AllAtomicNumbers,
        "atomic/avg_num_neighbours": AvgNumNeighbours,
        "atomic/force_std": ForceStd,
        # "atomic/energy_per_atom_lstsq": EnergyPerAtomLstsq,
    }
)
