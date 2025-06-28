from collections.abc import Hashable, Sequence
import logging
from typing import TYPE_CHECKING, Optional, Union

import e3nn_jax as e3j
from flax import linen
import jax
import jax.lax
import jax.numpy as jnp
import jraph
from pytray import tree

from . import _base, _experimental, utils

if TYPE_CHECKING:
    from tensorial import gcnn

_LOGGER = logging.getLogger(__name__)

__all__ = "Rescale", "IndexedLinear", "IndexedRescale"


class Rescale(linen.Module):
    """
    Applies constant rescaling and/or shifting to fields in a `jraph.GraphsTuple`.

    This module modifies specified fields in the graph — which may be located in the
    `nodes`, `edges`, or `globals` — by multiplying them with a scalar factor (`scale`)
    and/or adding a constant offset (`shift`). This is useful for normalizing or denormalizing
    values, or applying consistent physical unit conversions.

    Both `scale_fields` and `shift_fields` may be either a single string (e.g. "nodes.energy")
    or a sequence of path strings. Missing fields are ignored silently.

    Example usage:
    --------------
    >>> Rescale(shift_fields='nodes.energy', shift=12.5)
    shifts the energy stored in each node by 12.5.

    >>> Rescale(scale_fields=['globals.volume'], scale=1e-3)
    rescales the global volume by 1e-3.

    Attributes:
    -----------
    shift_fields : Union[str, Sequence[Hashable]]
        Path(s) to the fields to which a constant shift should be applied.

    scale_fields : Union[str, Sequence[Hashable]]
        Path(s) to the fields to which a constant scale should be applied.

    shift : jax.Array
        Scalar constant to be added to all values in `shift_fields`. Defaults to 0.0.

    scale : jax.Array
        Scalar constant to multiply all values in `scale_fields`. Defaults to 1.0.

    Notes:
    ------
    - Fields that are not found in the graph are skipped silently.
    - If a global field is shifted, a warning is logged that the field will no longer
      be size extensive with respect to the number of nodes or edges.
    """

    shift_fields: Union[str, Sequence[Hashable]] = tuple()
    scale_fields: Union[str, Sequence[Hashable]] = tuple()
    shift: jax.Array = 0.0
    scale: jax.Array = 1.0

    def setup(self):
        # pylint: disable=attribute-defined-outside-init
        shift_fields = (
            self.shift_fields if not isinstance(self.shift_fields, str) else [self.shift_fields]
        )
        scale_fields = (
            self.scale_fields if not isinstance(self.scale_fields, str) else [self.scale_fields]
        )

        self._shift_fields = tuple(map(utils.path_from_str, shift_fields))
        self._scale_fields = tuple(map(utils.path_from_str, scale_fields))

        if self.shift != 0.0:
            for path in self._shift_fields:
                if path[0] == "globals":
                    _LOGGER.info(
                        "Setting shift `%s` to `%f`, this field will no longer be size "
                        "extensive with the number of nodes/edges",
                        utils.path_to_str(path),
                        self.shift,
                    )

    @linen.compact
    @_base.shape_check
    def __call__(
        self, graph: jraph.GraphsTuple
    ) -> jraph.GraphsTuple:  # pylint: disable=arguments-differ
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
    """
    Applies a per-type affine transformation (scale and shift) to a specified field in a graph.

    Each input is scaled and shifted based on an associated index (e.g. atomic or node type).
    The transformation is of the form: `output = input * scale + shift`, where both `scale` and
    `shift` are either learnable parameters or provided constants, indexed by the value in
    `index_field`.

    This is typically used to normalize or denormalize features like node energies, depending on
    the type of node or atom.

    Attributes:
        num_types (int): Number of unique types (i.e. distinct values in `index_field`). Determines
            the number of learnable `scale` and `shift` parameters.
        index_field (str): Path (e.g. "nodes.type") to the array of indices used to select the
            scale and shift for each input.
        field (str): Path to the input field to be rescaled.
        out_field (Optional[str]): Path to the output field. If `None`, the result is written to
            `field`.
        shifts (Optional[ArrayLike]): Optional constant shift values of shape `(num_types,)`. If
            `None`, the shifts are learned parameters initialized with `shift_init`.
        scales (Optional[ArrayLike]): Optional constant scale values of shape `(num_types, 1)`. If
            `None`, the scales are learned parameters initialized with `rescale_init`.
        rescale_init (Initializer): Initializer for learnable scale parameters.
        shift_init (Initializer): Initializer for learnable shift parameters.

    Returns:
        jraph.GraphsTuple: A new graph with the specified field transformed and stored at
            `out_field`.

    Raises:
        ValueError: If the number of types does not match the shape of provided `scales` or
            `shifts`.

    Notes:
        - Supports `e3nn_jax.IrrepsArray` input and preserves irreps metadata.
        - Uses `jax.vmap` internally for efficiency across nodes.
    """

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
        self._out_field = (
            self._field if self.out_field is None else utils.path_from_str(self.out_field)
        )

        self._scales = (
            self.param(
                "scales",
                self.rescale_init,
                (self.num_types, 1),
            )
            if self.scales is None
            else self._to_array(self.scales, self.num_types)
        )

        self._shifts = (
            self.param("shifts", self.shift_init, (self.num_types,))
            if self.shifts is None
            else self._to_array(self.shifts, self.num_types)
        )

        # assert self._scales.shape == self._shifts.shape

    @_base.shape_check
    def __call__(self, graph: jraph.GraphsTuple):
        graph_dict: dict = graph._asdict()

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
        outs = jax.vmap(lambda inp, scale, shift: inp * scale + shift, (0, 0, 0))(
            inputs, scales, shifts
        )
        if output_irreps is not None:
            outs = e3j.IrrepsArray(output_irreps, outs)

        return _experimental.update_graph(graph).set(self._out_field, outs).get()

    @staticmethod
    def _to_array(value, num_types):
        return value if isinstance(value, jax.Array) else jnp.array([value] * num_types)


class IndexedLinear(linen.Module):
    """
    Applies an indexed linear transformation to a field in a `GraphsTuple`.

    This module performs a linear transformation on a per-element basis, where each element is
    routed through a specific linear layer determined by an associated index array. A separate
    set of learnable weights is maintained for each index value.

    Attributes:
        irreps_out (Union[str, e3j.Irreps]): The output irreducible representations of the linear
            transformation.
        num_types (int): Number of distinct index values, corresponding to the number of weight
            sets.
        index_field (str): Dot-separated path to the index array within the `GraphsTuple`.
        field (str): Dot-separated path to the input features within the `GraphsTuple`.
        out_field (Optional[str]): Dot-separated path where output features should be written. If
            None, overwrites `field`. name (str): Optional name for the internal Linear module.

    Args:
        graph (jraph.GraphsTuple): A graph with fields specified by `index_field` and `field`.

    Returns:
        jraph.GraphsTuple: A new graph with updated features at `out_field`, where each input vector
        has been transformed by a linear layer corresponding to its associated index.

    Raises:
        KeyError: If the specified `field` or `index_field` does not exist in the graph.
        ValueError: If the index values exceed the range `[0, num_types - 1]`.

    Example:
        If `graph.nodes` contains input features and `graph.nodes["type"]` contains integer indices
        in `[0, num_types)`, the module applies a learned linear map per type:

            IndexedLinear("64x0e", num_types=5, index_field="nodes.type", field="nodes.feat")

        Each node's "feat" will be transformed by a different `Linear` layer according to its
        "type".
    """

    irreps_out: Union[str, e3j.Irreps]
    num_types: int
    index_field: str
    field: str
    out_field: Optional[str] = None
    name: str = None

    @linen.compact
    @_base.shape_check
    def __call__(
        self, graph: jraph.GraphsTuple
    ) -> jraph.GraphsTuple:  # pylint: disable=arguments-differ
        index_field = utils.path_from_str(self.index_field)
        field: "gcnn.TreePath" = utils.path_from_str(self.field)
        out_field = field if self.out_field is None else utils.path_from_str(self.out_field)
        linear = e3j.flax.Linear(
            self.irreps_out,
            num_indexed_weights=self.num_types,
            name=self.name,
            force_irreps_out=True,
        )

        graph_dict = graph._asdict()

        # Get the indexes and values
        indexes = tree.get_by_path(graph_dict, index_field)
        inputs = tree.get_by_path(graph_dict, field)

        # Call the branches and update the graph
        outs = linear(indexes, inputs)
        return _experimental.update_graph(graph).set(out_field, outs).get()
