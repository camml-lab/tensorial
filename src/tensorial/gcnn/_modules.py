# -*- coding: utf-8 -*-
from typing import Hashable, Sequence, Union

from flax import linen
import jax
import jraph
from pytray import tree

from . import utils

__all__ = ('Rescale',)


class Rescale(linen.Module):
    """Rescale and shift any attributes of a graph by constants.  This can be applied to either nodes, edges, or globals
    by specifying the shift_fields and scale_fields e.g.:

        Rescale(shift_fields='nodes.energy', shift=12.5)

    will shift the energy attribute of the nodes by 12.5.
    Note that if the field is missing from the graph, then this module will ignore it.
    """
    shift_fields: Union[str, Sequence[Hashable]] = tuple()
    scale_fields: Union[str, Sequence[Hashable]] = tuple()
    shift: jax.Array = 0.0
    scale: jax.Array = 1.0

    def setup(self):
        # pylint: disable=attribute-defined-outside-init
        shift_fields = (self.shift_fields if not isinstance(self.shift_fields, str) else [self.shift_fields])
        scale_fields = (self.scale_fields if not isinstance(self.scale_fields, str) else [self.scale_fields])

        self._shift_fields = tuple(map(utils.path_from_str, shift_fields))
        self._scale_fields = tuple(map(utils.path_from_str, scale_fields))

    @linen.compact
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:  # pylint: disable=arguments-differ
        graph_dict = utils.UpdateDict(graph._asdict())

        # Scale first
        for field in self._scale_fields:
            try:
                new_value = tree.get_by_path(graph_dict, field) * self.scale
                tree.set_by_path(graph_dict, field, new_value)
            except KeyError:
                pass  # Ignore missing keys

        # Now shift
        for field in self._shift_fields:
            try:
                new_value = tree.get_by_path(graph_dict, field) + self.shift
                tree.set_by_path(graph_dict, field, new_value)
            except KeyError:
                pass  # Ignore missing keys

        return jraph.GraphsTuple(**graph_dict._asdict())
