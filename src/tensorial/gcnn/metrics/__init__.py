"""Graph metrics for :mod:`tensorial.gcnn` models.

Metrics are aggregated over a graph batch (see
:class:`tensorial.gcnn.metrics._base.GraphMetric`) and can be split into
per-component contributions.
"""

from . import _base, _contributions
from ._base import *
from ._contributions import *

__all__ = _base.__all__ + _contributions.__all__
