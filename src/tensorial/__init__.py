# -*- coding: utf-8 -*-
"""Library for machine learning on physical tensors"""

from . import base, config, metrics, tensors
from .base import *
from .tensors import *

__version__ = '0.2.0'

__all__ = base.__all__ + tensors.__all__ + ('config', 'metrics')
