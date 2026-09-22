"""Library for machine learning on physical tensors"""

from . import (
    base,
    config,
    datasets,
    gcnn,
    geometry,
    reaxkit,
    tensors,
    typing,
    utils,
)
from .base import *
from .reaxkit import ReaxModule
from .tensors import *

__version__ = "0.6.5"

__all__ = (
    base.__all__
    + tensors.__all__
    + (
        "datasets",
        "config",
        "gcnn",
        "geometry",
        "typing",
        "ReaxModule",
        "utils",
    )
)
