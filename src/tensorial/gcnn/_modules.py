# -*- coding: utf-8 -*-
import logging
from typing import Hashable, Optional, Sequence, Union

import e3nn_jax as e3j
from flax import linen
import jax
import jax.lax
import jax.numpy as jnp
import jraph
from pytray import tree

from . import utils

_LOGGER = logging.getLogger(__name__)

__all__ = 'Rescale', 'IndexedLinear', 'IndexedRescale'


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

        if self.shift != 0.0:
            for path in self._shift_fields:
                if path[0] == 'globals':
                    _LOGGER.warning(
                        'Setting global shift %s to %d, this field will no longer be size extensive with '
                        'the number of nodes/edges',
                        utils.path_to_str(path),
                        self.shift,
                    )

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


class IndexedRescale(linen.Module):
    num_types: int
    index_field: str
    field: str
    out_field: Optional[str] = None
    shifts: Optional[jax.typing.ArrayLike] = None
    scales: Optional[jax.typing.ArrayLike] = None

    rescale_init: linen.initializers.Initializer = linen.initializers.lecun_normal()
    shift_init: linen.initializers.Initializer = linen.initializers.zeros_init()

    def setup(self):
        # pylint: disable=attribute-defined-outside-init
        self._index_field = utils.path_from_str(self.index_field)
        self._field = utils.path_from_str(self.field)
        self._out_field = self._field if self.out_field is None else utils.path_from_str(self.out_field)

        self._scales = self.param(
            'scales',
            self.rescale_init,
            (self.num_types, 1),
        ) if self.scales is None else self._to_array(self.scales, self.num_types)

        self._shifts = self.param('shifts', self.shift_init,
                                  (self.num_types,
                                   )) if self.shifts is None else self._to_array(self.shifts, self.num_types)

        # assert self._scales.shape == self._shifts.shape

    def __call__(self, graph: jraph.GraphsTuple):
        graph_dict = utils.UpdateDict(graph._asdict())

        # Get the indexes and values
        indexes = tree.get_by_path(graph_dict, self._index_field)
        inputs = tree.get_by_path(graph_dict, self._field)
        if isinstance(inputs, e3j.IrrepsArray):
            output_irreps = inputs.irreps
            inputs = inputs.array
        else:
            output_irreps = None

        # Get the shifts and scales
        scales = jnp.take(self._scales, indexes)
        shifts = jnp.take(self._shifts, indexes)
        outs = jax.vmap(lambda inp, scale, shift: inp * scale + shift, (0, 0, 0))(inputs, scales, shifts)
        if output_irreps is not None:
            outs = e3j.IrrepsArray(output_irreps, outs)

        tree.set_by_path(graph_dict, self._out_field, outs)
        return graph._replace(**graph_dict._asdict())

    @staticmethod
    def _to_array(value, num_types):
        return value if isinstance(value, jax.Array) else jnp.array([value] * num_types)


class IndexedLinear(linen.Module):
    """
    Applies a linear transform to a an array of values where a separate index array determines which linear layer the
    value gets passed to.  Weights are per index.
    """

    irreps_out: Union[str, e3j.Irreps]
    num_types: int
    index_field: str
    field: str
    out_field: Optional[str] = None
    name: str = None

    @linen.compact
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:  # pylint: disable=arguments-differ
        index_field = utils.path_from_str(self.index_field)
        field = utils.path_from_str(self.field)
        out_field = field if self.out_field is None else utils.path_from_str(self.out_field)
        linear = e3j.flax.Linear(
            self.irreps_out,
            num_indexed_weights=self.num_types,
            name=self.name,
            force_irreps_out=True,
        )

        graph_dict = utils.UpdateDict(graph._asdict())

        # Get the indexes and values
        indexes = tree.get_by_path(graph_dict, index_field)
        inputs = tree.get_by_path(graph_dict, field)

        # Call the branches and update the graph
        outs = linear(indexes, inputs)
        tree.set_by_path(graph_dict, out_field, outs)

        return graph._replace(**graph_dict._asdict())
