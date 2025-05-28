from collections.abc import Sequence
from typing import Union

import beartype
import equinox
import jax
import jax.numpy as jnp
import jaxtyping as jt
import jraph

from . import keys
from .. import _modules as gcnn_modules
from .. import keys as gcnn_keys
from ... import nn_utils, typing

__all__ = "SpeciesTransform", "per_species_rescale"


@jt.jaxtyped(typechecker=beartype.beartype)
class SpeciesTransform(equinox.Module):
    """
    Take an ordered list of species and transform them into an integer corresponding to their
    position in the list
    """

    atomic_numbers: jt.Int[jax.Array, "numbers"]
    field: str = keys.ATOMIC_NUMBERS
    out_field: str = gcnn_keys.SPECIES

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        atomic_numbers: Union[Sequence[int], jt.Int[typing.ArrayType, "numbers"]],
        field: str = keys.ATOMIC_NUMBERS,
        out_field: str = gcnn_keys.SPECIES,
    ):
        self.atomic_numbers = jnp.asarray(atomic_numbers)
        self.field = field
        self.out_field = out_field

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __call__(
        self, graph: jraph.GraphsTuple
    ) -> jraph.GraphsTuple:  # pylint: disable=arguments-differ
        nodes = graph.nodes
        nodes[self.out_field] = nn_utils.vwhere(nodes[self.field], self.atomic_numbers)

        return graph._replace(nodes=nodes)


def per_species_rescale(
    num_types: int,
    field: str,
    *,
    types_field: str = None,
    out_field: str = None,
    shifts: jax.typing.ArrayLike = None,
    scales: jax.typing.ArrayLike = None,
) -> gcnn_modules.IndexedRescale:
    types_field = types_field or ("nodes", gcnn_keys.SPECIES)
    return gcnn_modules.IndexedRescale(
        num_types,
        index_field=types_field,
        field=field,
        out_field=out_field,
        shifts=shifts,
        scales=scales,
    )
