# -*- coding: utf-8 -*-
"""Library for machine learning on physical tensors"""

from tensorial.metrics import metric

from . import base, config, metrics, tensorboards, tensors, training
from .base import *
from .tensors import *
from .training import *

__version__ = '0.2.0'

__all__ = base.__all__ + tensors.__all__ + training.__all__ + (
    'config', 'metrics', 'training', 'metric', 'tensorboards'
)
