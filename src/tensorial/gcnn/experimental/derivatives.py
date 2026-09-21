"""Back-compat shim re-exporting the graph differentiation helpers from ``.._diff``."""

from .._diff import GraphEntrySpec, MultiDerivative, SingleDerivative, diff

__all__ = "diff", "SingleDerivative", "MultiDerivative", "GraphEntrySpec"
