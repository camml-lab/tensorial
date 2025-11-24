from collections.abc import Hashable, Iterable, Mapping, MutableMapping
import numbers
from typing import TYPE_CHECKING

import jraph
import numpy as np

from tensorial.typing import Array, CellType, PbcType

from . import keys
from .. import _spatial as gcnn_graphs
from ... import base

if TYPE_CHECKING:
    try:
        import ase
    except ImportError:
        pass

    try:
        import pymatgen
    except ImportError:
        pass

__all__ = "graph_from_pymatgen", "graph_from_ase"


# too slow: @jt.jaxtyped(typechecker=beartype.beartype)
def graph_from_pymatgen(
    pymatgen_structure: "pymatgen.core.SiteCollection",
    r_max: numbers.Number,
    *,
    key_mapping: dict[str, str] | None = None,
    atom_include_keys: Iterable | None = ("numbers",),
    edge_include_keys: Iterable | None = tuple(),
    global_include_keys: Iterable | None = tuple(),
    cell: CellType | None = None,
    pbc: bool | PbcType | None = None,
    graph_globals: dict[str, Array] | None = None,
    **kwargs,
) -> jraph.GraphsTuple:
    """Create a jraph Graph from a pymatgen SiteCollection object or subclass
    (e.g. Structure, Molecule)

    Note that the special atom key "numbers" is used to retrieve atomic numbers using
    SiteCollection.atomic_numbers.
    All other keys are used to retrieve site properties using SiteCollection.site_properties.

    Args:
        pymatgen_structure: the SiteCollection object
        r_max: the maximum neighbour distance to use when considering
            two atoms to be neighbours
        key_mapping
        atom_include_keys
        global_include_keys
        cell: an optional unit cell (otherwise will be taken from
            Structure.lattice.matrix if it exists)
        pbc: an optional periodic boundary conditions array [bool, bool,
            bool] (otherwise will be taken from Structure.lattice.pbc if
            it exists)

    Returns:
        the atomic graph
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

    graph_globals = graph_globals or {}
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
    key_mapping: dict[str, str] | None = None,
    atom_include_keys: Iterable | None = ("numbers",),
    edge_include_keys: Iterable | None = tuple(),
    global_include_keys: Iterable | None = tuple(),
    cell: CellType | None = None,
    pbc: bool | PbcType | None = None,
    use_calculator: bool = True,
    **kwargs,
) -> jraph.GraphsTuple:
    """Create a jraph Graph from an ase.Atoms object

    Args:
        ase_atoms: the Atoms object
        r_max: the maximum neighbour distance to use when considering
            two atoms to be neighbours
        key_mapping
        atom_include_keys
        global_include_keys
        cell: an optional unit cell (otherwise will be taken from
            ase.cell)
        pbc: an optional periodic boundary conditions array [bool, bool,
            bool] (otherwise will be taken from ase.pbc)
        use_calculator: if `True`, will try to use an attached
            calculator get additional properties

    Returns:
        the atomic graph
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
