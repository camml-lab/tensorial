"""Datasets built on top of :mod:`tensorial.gcnn.data`.

This package simply re-exports the public API so users can import as
``tensorial.datasets.<name>``.
"""

from . import qm9
from .qm9 import *

__all__ = qm9.__all__
