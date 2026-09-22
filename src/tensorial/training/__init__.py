"""Backward-compatible namespace for :class:`tensorial.reaxkit.ReaxModule`.

Deprecated: use :mod:`tensorial.reaxkit` directly. This is a re-export
shim kept only for backward compatibility and will be removed in a
future release.
"""

import warnings

from . import _module
from ._module import ReaxModule

warnings.warn(
    "tensorial.training is deprecated and will be removed in a future version. "
    "Import ReaxModule from tensorial.reaxkit instead.",
    category=DeprecationWarning,
    stacklevel=2,
)

__all__ = _module.__all__
