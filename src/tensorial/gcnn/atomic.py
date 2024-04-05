# -*- coding: utf-8 -*-
from typing import Any, Dict, Hashable, Iterable, List, Mapping, MutableMapping, Optional, Union

import e3nn_jax as e3j
from flax import linen
import jax
import jax.numpy as jnp
import jraph

from tensorial import base

from . import _graphs

ENERGY_PER_ATOM = 'energy/atom'
TOTAL_ENERGY = 'total_energy'
FORCES = 'forces'
STRESS = 'stress'
VIRIAL = 'virial'
PBC = 'pbc'
ATOMIC_NUMBERS = 'atomic_numbers'
ATOMIC_TYPE_IDX = 'atomic_type_idx'

# Global quantities
ASE_GLOBAL_KEYS = {'energy', 'free_energy', 'stress', 'magmom'}
# Per-atom quantities
ASE_ATOM_KEYS = {'numbers', 'forces', 'stresses', 'charges', 'magmoms', 'energies'}

PyTree = Any


def graph_from_ase(
    ase_atoms: 'ase.atoms.Atoms',
    r_max: float,
    key_mapping: Optional[Dict[str, str]] = None,
    atom_include_keys: Optional[Iterable] = ('numbers',),
    global_include_keys: Optional[Iterable] = tuple(),
    cell: jax.Array = None,
    pbc: jax.Array = None,
) -> jraph.GraphsTuple:
    """
    Create a jraph Graph from an ase.Atoms object

    :param ase_atoms: the Atoms object
    :param r_max: the maximum neighbour distance to use when considering two atoms to be neighbours
    :param key_mapping:
    :param atom_include_keys:
    :param global_include_keys:
    :param cell: an optional unit cell (otherwise will be taken from ase.cell)
    :param pbc: an optional periodic boundary conditions array [bool, bool, bool] (otherwise will be taken from ase.pbc)
    :return:
    """
    from ase.calculators import singlepoint
    import ase.stress

    key_mapping = key_mapping or {}
    _key_mapping = {
        'forces': FORCES,
        'energy': TOTAL_ENERGY,
        'numbers': ATOMIC_NUMBERS,
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

    if ase_atoms.calc is not None:
        if not isinstance(
            ase_atoms.calc,
            (singlepoint.SinglePointCalculator, singlepoint.SinglePointDFTCalculator),
        ):
            raise NotImplementedError(f'`from_ase` does not support calculator {type(ase_atoms.calc).__name__}')

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
                raise RuntimeError(f'Unexpected shape for {key}, got: {graph_globals[key].shape}')

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
        graph_globals=graph_globals,
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


class SpeciesTransform(linen.Module):
    """Take an ordered list of species and transform them into an integer corresponding to their position in the list"""

    atomic_numbers: List[int]
    field: str = ATOMIC_NUMBERS
    out_field: str = ATOMIC_TYPE_IDX

    def setup(self):
        self._atomic_numbers = jnp.array(self.atomic_numbers)  # pylint: disable=attribute-defined-outside-init

    def __call__(self, graph: jraph.GraphsTuple):  # pylint: disable=arguments-differ
        vwhere = jax.vmap(lambda num: jnp.argwhere(num == self._atomic_numbers, size=1)[0])
        nodes = graph.nodes
        nodes[self.out_field] = vwhere(nodes[self.field])[:, 0]
        return graph._replace(nodes=nodes)


class PerSpeciesRescale(linen.Module):
    shifts: jax.Array
    scales: jax.Array
    field: str = ENERGY_PER_ATOM
    out_field: Optional[str] = None
    scales_trainable: bool = False
    shifts_trainable: bool = False
    atomic_type_field: str = ATOMIC_TYPE_IDX

    @staticmethod
    def construct(
        shifts: Union[jax.Array, float],
        scales: Union[jax.Array, float],
        n_types: int = None,
        field: str = ENERGY_PER_ATOM,
        out_field: Optional[str] = None,
        scales_trainable: bool = False,
        shifts_trainable: bool = False,
        atomic_type_field: str = ATOMIC_TYPE_IDX,
    ):
        if (isinstance(shifts, float) or isinstance(scales, float)) and n_types is None:
            raise ValueError('If shifts or scales is a scalar, the number of types must be specified')

        if isinstance(shifts, float):
            shifts = jnp.array([shifts] * n_types)
        if isinstance(scales, float):
            scales = jnp.array([scales] * n_types)
        out_field = out_field or field

        return PerSpeciesRescale(
            shifts, scales, field, out_field, scales_trainable, shifts_trainable, atomic_type_field
        )

    @linen.compact
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:  # pylint: disable=arguments-differ
        nodes = graph.nodes
        species_idx = nodes[self.atomic_type_field]
        shifts = self.shifts[species_idx]
        scales = self.scales[species_idx]
        rescaling: e3j.IrrepsArray = nodes[self.field]

        # Vectorise scaling and shifting
        vmul = jax.vmap(jnp.multiply, (0, 0))
        vadd = jax.vmap(jnp.add, (0, 0))

        rescaled = vadd(vmul(rescaling.array, scales), shifts)
        nodes[self.out_field] = e3j.IrrepsArray(rescaling.irreps, rescaled)
        return graph._replace(nodes=nodes)
