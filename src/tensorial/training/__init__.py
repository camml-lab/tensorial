"""Backward-compatible namespace for :class:`tensorial.reaxkit.ReaxModule`.

New code should import from :mod:`tensorial.reaxkit` directly.
"""

from . import _module
from ._module import *

__all__ = _module.__all__
