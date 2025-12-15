"""Common utility functions that operate on graphs"""

from . import graph_ops

__all__ = ("reduce",)


def reduce(*args, **kwargs):
    return graph_ops.graph_segment_reduce(*args, **kwargs)
