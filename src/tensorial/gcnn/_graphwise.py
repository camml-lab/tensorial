# -*- coding: utf-8 -*-
"""Modules that act on the entire graph or global data"""
from typing import Optional

import e3nn_jax as e3j
from flax import linen
import jax
import jraph

__all__ = ('GlobalRescale',)


class GlobalRescale(linen.Module):
    field: str
    out_field: str
    shift: jax.Array = 0.
    scale: jax.Array = 1.
    scales_trainable: bool = False
    shifts_trainable: bool = False

    @staticmethod
    def construct(
        field: str,
        shift: jax.Array = 0.,
        scale: jax.Array = 1.,
        out_field: Optional[str] = None,
        scales_trainable: bool = False,
        shifts_trainable: bool = False,
    ) -> 'GlobalRescale':
        out_field = out_field or field
        return GlobalRescale(
            field=field,
            out_field=out_field,
            shift=shift,
            scale=scale,
            scales_trainable=scales_trainable,
            shifts_trainable=shifts_trainable,
        )

    @linen.compact
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:  # pylint: disable=arguments-differ
        globals_ = graph.globals
        rescaling: e3j.IrrepsArray = globals_[self.field]
        globals_[self.out_field] = rescaling * self.scale + self.shift
        return graph._replace(globals=globals_)
