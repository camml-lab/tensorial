"""
This module exposes some commonly used metrics
"""

import e3nn_jax
import reax.metrics

from . import keys
from .. import gcnn

__all__ = (
    "EnergyPerAtomRmse",
    "EnergyPerAtomMae",
    "ForceRmse",
    "StressRmse",
)


def _convert(*args, mask=None) -> tuple:
    """Convert irreps arrays to regular arryas as not all metric operations will work on irrep
    arrays"""
    vals = tuple(
        entry.array if isinstance(entry, e3nn_jax.IrrepsArray) else entry for entry in args
    )
    if mask is not None:
        mask = mask.array if isinstance(mask, e3nn_jax.IrrepsArray) else mask
        vals = (*vals, mask)

    return vals


# pylint: disable=assignment-from-no-return

EnergyPerAtomRmse = gcnn.graph_metric(
    reax.metrics.RootMeanSquareError.from_fun(_convert),
    targets="globals.energy",
    predictions="globals.predicted_energy",
    mask="globals.mask",
    normalise_by="n_node",
)

EnergyPerAtomMae = gcnn.graph_metric(
    reax.metrics.MeanAbsoluteError.from_fun(_convert),
    targets="globals.energy",
    predictions="globals.predicted_energy",
    mask="globals.mask",
    normalise_by="n_node",
)

ForceRmse = gcnn.graph_metric(
    reax.metrics.RootMeanSquareError.from_fun(_convert),
    targets="nodes.forces",
    predictions="nodes.predicted_forces",
    mask="nodes.mask",
)

StressRmse = gcnn.graph_metric(
    reax.metrics.RootMeanSquareError.from_fun(_convert)(),
    targets=f"globals.{keys.STRESS}",
    predictions=f"globals.{keys.predicted(keys.STRESS)}",
    mask=f"globals.{keys.MASK}",
)
