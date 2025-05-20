"""Library for machine learning on physical tensors"""

from tensorial.metrics import metric

from . import base, config, data, geometry, metrics, tensors, training, typing, utils
from .base import *
from .tensors import *
from .training import *
from .training import ReaxModule

__version__ = "0.3.0"

__all__ = (
    base.__all__
    + tensors.__all__
    + training.__all__
    + (
        "config",
        "geometry",
        "metrics",
        "training",
        "metric",
        "data",
        "typing",
        "ReaxModule",
        "utils",
    )
)
