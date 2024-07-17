from collections.abc import Sequence
import functools
import random

import ase.build
import ase.calculators.singlepoint
import ase.collections
import jraph
import numpy as np
import pytest

from tensorial.gcnn import atomic


@pytest.fixture
def molecule_dataset() -> Sequence[jraph.GraphsTuple]:
    """Generate some fake data using a collection of molecules from ASE"""
    r_max = 3.0

    dataset = []
    for molecule, _ in zip(ase.collections.g2, range(10)):
        molecule.arrays[atomic.TOTAL_ENERGY] = random.random()
        molecule.arrays[atomic.FORCES] = np.random.rand(len(molecule), 3)
        dataset.append(molecule)

    return list(
        map(
            functools.partial(
                atomic.graph_from_ase,
                r_max=r_max,
                atom_include_keys=("numbers", atomic.FORCES),
                global_include_keys=(atomic.TOTAL_ENERGY,),
            ),
            dataset,
        )
    )
