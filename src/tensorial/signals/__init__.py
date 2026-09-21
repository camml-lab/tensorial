"""Signal-space machinery: bases, functions, and expansion.

Public sub-modules:

- :mod:`.functions` — abstract signal values (Deltas, Gaussians, sums).
- :mod:`.radials` — radial basis functions.
- :mod:`.bases` — spherical and radial-spherical bases.
- :mod:`.expansion` — the :func:`.expansion.expand` singledispatch helper.
"""

from . import functions

__all__ = ("functions",)
