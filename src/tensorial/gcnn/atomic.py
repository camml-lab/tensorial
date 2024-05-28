# -*- coding: utf-8 -*-
import numbers
from typing import Any, Hashable, Iterable, Mapping, MutableMapping, Optional, Sequence

import beartype
import equinox
import jax
import jax.numpy as jnp
import jaxtyping as jt
import jraph

from tensorial import base

from . import _graphs, _modules, keys

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


@jt.jaxtyped(beartype.beartype)
def graph_from_ase(
    ase_atoms: "ase.atoms.Atoms",
    r_max: numbers.Number,
    key_mapping: Optional[dict[str, str]] = None,
    atom_include_keys: Optional[Iterable] = ("numbers",),
    edge_include_keys: Optional[Iterable] = tuple(),
    global_include_keys: Optional[Iterable] = tuple(),
    cell: Optional[jt.Float[jax.typing.ArrayLike, "3 3"]] = None,
    pbc: Optional[bool | _graphs.PbcType] = None,
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


@jt.jaxtyped(beartype.beartype)
class SpeciesTransform(equinox.Module):
    """
    Take an ordered list of species and transform them into an integer corresponding to their
    position in the list
    """

    atomic_numbers: jt.Int[jax.Array, "numbers"]
    field: str = ATOMIC_NUMBERS
    out_field: str = keys.SPECIES

    def __init__(self, atomic_numbers: Sequence[int], field=ATOMIC_NUMBERS, out_field=keys.SPECIES):
        self.atomic_numbers = jnp.asarray(
            atomic_numbers
        )  # pylint: disable=attribute-defined-outside-init
        self.field = field
        self.out_field = out_field

    def __call__(self, graph: jraph.GraphsTuple):  # pylint: disable=arguments-differ
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
