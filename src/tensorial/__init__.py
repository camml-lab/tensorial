"""Library for machine learning on physical tensors"""

# flake8: noqa: F402
# pylint: disable=wrong-import-position
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


from . import (
    base,
    config,
    datasets,
    gcnn,
    geometry,
    tensors,
    training,
    typing,
    utils,
)
from .base import *
from .tensors import *
from .training import *
from .training import ReaxModule

__version__ = "0.6.2"

__all__ = (
    base.__all__
    + tensors.__all__
    + training.__all__
    + (
        "datasets",
        "config",
        "gcnn",
        "geometry",
        "training",
        "typing",
        "ReaxModule",
        "utils",
    )
)
