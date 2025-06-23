from collections.abc import Hashable, Iterable
import numbers
from typing import Mapping, MutableMapping, Optional, Union

import jraph
import numpy as np

from . import keys
from .. import _spatial as gcnn_graphs
from ... import base, typing

__all__ = "graph_from_pymatgen", "graph_from_ase"


# too slow: @jt.jaxtyped(typechecker=beartype.beartype)
def graph_from_pymatgen(
    pymatgen_structure: "pymatgen.SiteCollection",
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
    """Create a jraph Graph from a pymatgen SiteCollection object or subclass
    (e.g. Structure, Molecule)

    Note that the special atom key "numbers" is used to retrieve atomic numbers using
    SiteCollection.atomic_numbers.
    All other keys are used to retrieve site properties using SiteCollection.site_properties.

    :param pymatgen_structure: the SiteCollection object
    :param r_max: the maximum neighbour distance to use when considering two atoms to be neighbours
    :param key_mapping:
    :param atom_include_keys:
    :param global_include_keys:
    :param cell: an optional unit cell (otherwise will be taken from Structure.lattice.matrix if
        it exists)
    :param pbc: an optional periodic boundary conditions array [bool, bool, bool] (otherwise will be
        taken from Structure.lattice.pbc if it exists)
    :return: the atomic graph
    """
    # pylint: disable=too-many-branches
    key_mapping = key_mapping or {}
    _key_mapping = {
        "forces": keys.FORCES,
        "energy": keys.TOTAL_ENERGY,
        "numbers": keys.ATOMIC_NUMBERS,
    }
    _key_mapping.update(key_mapping)
    key_mapping = _key_mapping
    del _key_mapping

    positions = pymatgen_structure.cart_coords
    if hasattr(pymatgen_structure, "lattice"):
        cell = cell or pymatgen_structure.lattice.matrix
        pbc = pbc or pymatgen_structure.lattice.pbc

    atoms = {}
    if "numbers" in atom_include_keys:
        atoms[key_mapping.get("numbers", "numbers")] = np.asarray(pymatgen_structure.atomic_numbers)
        atom_include_keys = set(atom_include_keys) - {"numbers"}
    for key in atom_include_keys:
        get_attrs(atoms, pymatgen_structure.site_properties, key, key_mapping)

    edges = {}
    for key in edge_include_keys:
        get_attrs(edges, pymatgen_structure.properties, key, key_mapping)

    graph_globals = {}
    for key in global_include_keys:
        get_attrs(graph_globals, pymatgen_structure.properties, key, key_mapping)

    return gcnn_graphs.graph_from_points(
        pos=positions,
        fractional_positions=False,
        r_max=r_max,
        cell=cell,
        pbc=pbc,
        nodes=atoms,
        edges=edges,
        graph_globals=graph_globals,
        **kwargs,
    )


# too slow: @jt.jaxtyped(typechecker=beartype.beartype)
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
    use_calculator: bool = True,
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
    :param use_calculator: if `True`, will try to use an attached calculator get additional
        properties
    :return: the atomic graph
    """
    # pylint: disable=too-many-branches
    from ase.calculators import singlepoint
    import ase.stress

    key_mapping = key_mapping or {}
    _key_mapping = {
        "forces": keys.FORCES,
        "energy": keys.TOTAL_ENERGY,
        "numbers": keys.ATOMIC_NUMBERS,
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

    if use_calculator and ase_atoms.calc is not None:
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
    for key in (keys.STRESS, keys.VIRIAL):
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

    atom_graph = gcnn_graphs.graph_from_points(
        pos=ase_atoms.positions,
        fractional_positions=False,
        r_max=r_max,
        cell=cell.__array__() if pbc.any() else None,
        pbc=pbc,
        nodes=atoms,
        edges=edges,
        graph_globals=graph_globals,
        **kwargs,
    )
    return atom_graph


def get_attrs(store_in: MutableMapping, get_from: Mapping, key: Hashable, key_map: Mapping) -> bool:
    out_key = key_map.get(key, key)
    try:
        value = get_from[key]
    except KeyError:
        # Couldn't find the attribute
        return False

    store_in[out_key] = value
    return True
