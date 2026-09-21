"""Core abstractions for describing tensorial types.

This module defines the building blocks that the rest of tensorial is built on
top of.  An *irrep attribute* (see :class:`Attr`) associates a single irreducible
representation (irrep) or irreps with a value, and knows how to turn a raw value
into an :class:`e3nn_jax.IrrepsArray` via :meth:`Attr.create_tensor` and back
via :meth:`Attr.from_tensor`.

An *irreps object* (see :class:`IrrepsObj`) aggregates a named collection of
attributes, so that a structured container of tensorial values can be represented
by a single, flattened :class:`e3nn_jax.IrrepsArray`.  The functions
:func:`create`, :func:`create_tensor`, :func:`from_tensor`, :func:`irreps` and
:func:`tensorial_attrs` provide a uniform, single-dispatch way to move between
these structured descriptions and flat tensors for any supported "tensorial"
descriptor (an :class:`Attr`, a :class:`IrrepsObj` subclass, a mapping, a frozen
dict, an :class:`e3nn_jax.Irreps`, or even a plain string).
"""

from collections.abc import Mapping
import functools
from typing import Any, Generic, TypeVar

import beartype
import e3nn_jax as e3j
import equinox
from flax import linen
import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
from reax.utils import arrays

from tensorial.typing import IntoIrreps

__all__ = (
    "IrrepsObj",
    "IrrepsTree",
    "Attr",
    "create",
    "create_tensor",
    "irreps",
    "get",
    "Tensorial",
    "tensorial_attrs",
    "from_tensor",
    "as_array",
)


Array = jax.typing.ArrayLike

T = TypeVar("T")


def atleast_1d(arr, np_=jnp) -> jax.Array | np.ndarray:
    """Return the input as an array with at least one dimension.

    Arrays that are already one-dimensional or higher are returned unchanged.
    Scalars (0-D arrays and scalars) are reshaped to a 1-D array of length 1.

    Args:
        arr: the value to convert (anything accepted by ``np_.asarray``)
        np_: the array backend to use. Defaults to the backend inferred from
            ``arr`` (see :func:`tensorial.utils.infer_backend`). Pass ``np`` or
            ``jnp`` explicitly if you need to force a specific backend.

    Returns:
        an array with ``ndim >= 1`` in the chosen backend
    """
    np_ = np_ if np_ is not None else arrays.infer_backend(arr)
    arr = np_.asarray(arr)
    return arr if np_.ndim(arr) >= 1 else np_.reshape(arr, -1)


def as_array(arr: jt.ArrayLike | e3j.IrrepsArray) -> jax.Array:
    """Get a standard JAX array given either:
        1. a numpy.ndarray
        2. an e3nn_jax.IrrepsArray, or
        3. a jax.Array (in which case it is returned unmodified)

    Args:
        arr: the array to get the value for

    Returns:
        the JAX array
    """
    if isinstance(arr, e3j.IrrepsArray):
        return arr.array

    return jnp.asarray(arr)


class Attr(equinox.Module, Generic[T]):
    """An attribute that carries a single irrep, plus the transforms that
    move values to and from tensor representation.

    Subclasses choose the irreps they represent and, if needed, override
    :meth:`create_tensor` and/or :meth:`from_tensor` to define how the raw
    input value ``T`` maps onto an :class:`e3nn_jax.IrrepsArray`.  The default
    implementations just wrap the (at-least-1d) value in an
    :class:`e3nn_jax.IrrepsArray` with the given irreps, unchanged.

    Attributes:
        irreps: the irreducible representation(s) carried by values created
            from this attribute.
    """

    irreps: e3j.Irreps

    def __init__(self, irreps: IntoIrreps) -> None:  # pylint: disable=redefined-outer-name
        self.irreps = e3j.Irreps(irreps)

    @jt.jaxtyped(typechecker=beartype.beartype)
    def create_tensor(self, value: T) -> e3j.IrrepsArray:
        """Convert the raw value into a tensor with :attr:`irreps`.

        The default implementation just wraps ``atleast_1d(value)`` into an
        :class:`e3nn_jax.IrrepsArray` with the same irreps.

        Args:
            value: the raw value, of the type this attribute expects.

        Returns:
            an :class:`e3nn_jax.IrrepsArray` with :attr:`irreps`.
        """
        return e3j.IrrepsArray(self.irreps, atleast_1d(value))

    @jt.jaxtyped(typechecker=beartype.beartype)
    def from_tensor(self, tensor: e3j.IrrepsArray) -> T:
        """This can be overwritten to perform the backward transform of `create_tensor`"""
        return tensor


class IrrepsObj:
    """An object that contains tensorial attributes.

    Subclasses describe a structured tensor type by declaring attributes whose
    values are themselves "tensorial" (an :class:`Attr`, a nested
    ``IrrepsObj`` subclass, an :class:`e3nn_jax.Irreps`, a plain irreps string,
    or a mapping of name-to-tensorial).  The full type is then the ordered
    concatenation of the irreps of its attributes, in declaration order.
    """


Tensorial = Attr | IrrepsObj | type(IrrepsObj) | dict | linen.FrozenDict | e3j.Irreps
IrrepsTree = IrrepsObj | dict[str, Tensorial]
ValueType = Any | list["ValueType"] | dict[str, "ValueType"]


@functools.singledispatch
def create(tensorial: Tensorial, value: Mapping):
    """Build a per-attribute value dictionary for an :class:`IrrepsObj` subclass.

    For each attribute of ``tensorial``, the corresponding value from
    ``value`` is resolved into a concrete tensor (via :func:`create_tensor`
    for leaves).  The result is a plain ``dict`` that maps each attribute
    name to its tensor.

    Args:
        tensorial: an :class:`IrrepsObj` subclass describing the desired
            tensor structure.
        value: a mapping from attribute name to the raw value for that
            attribute.

    Returns:
        a ``dict`` mapping each attribute name to its tensor.

    Raises:
        TypeError: if ``tensorial`` is not an :class:`IrrepsObj` subclass.
    """
    if not issubclass(tensorial, IrrepsObj):
        raise TypeError(tensorial.__class__.__name__)

    value_dict = {}
    for name, val in tensorial_attrs(tensorial).items():
        value_dict[name] = create(val, value[name])

    return value_dict


@create.register
def _(attr: Attr, value) -> e3j.IrrepsArray:
    """Leaf: delegate to :func:`create_tensor`"""
    return create_tensor(attr, value)


@create.register
def _(attr: IrrepsObj, value) -> e3j.IrrepsArray:
    """Leaf, so create the tensor"""
    return create_tensor(attr, value)


@create.register
def _(attr: e3j.Irreps, value) -> e3j.IrrepsArray:
    """Leaf, so create the tensor"""
    return create_tensor(attr, value)


@functools.singledispatch
def irreps(tensorial: Tensorial) -> e3j.Irreps:
    """Return the total irreps carried by a tensorial descriptor.

    For an :class:`IrrepsObj` subclass this is the ordered sum of the irreps
    of each of its attributes, in declaration order.  Leaf descriptors
    (:class:`Attr` and :class:`e3nn_jax.Irreps`) return their own irreps.

    Args:
        tensorial: an :class:`IrrepsObj` subclass, an :class:`Attr`, or an
            :class:`e3nn_jax.Irreps`.

    Returns:
        the resulting :class:`e3nn_jax.Irreps`.

    Raises:
        TypeError: if ``tensorial`` is not a recognised tensorial descriptor.
    """
    if not issubclass(tensorial, IrrepsObj):
        raise TypeError(tensorial.__class__.__name__)

    # IrrepsObj code:
    total_irreps = None

    for name, val in tensorial_attrs(tensorial).items():
        try:
            total_irreps = val.irreps if total_irreps is None else total_irreps + val.irreps
        except AttributeError as exc:
            raise AttributeError(f"Failed to get irreps for {name}") from exc

    return total_irreps


@irreps.register
def _irreps_attr(attr: Attr) -> e3j.Irreps:
    """Attribute leaf: return its own irreps"""
    return attr.irreps


@irreps.register
def _irreps_irreps(tensorial: e3j.Irreps) -> e3j.Irreps:
    """Pure irreps leaf: return the irreps themselves"""
    return tensorial


@functools.singledispatch
def create_tensor(tensorial: Tensorial, value: ValueType) -> e3j.IrrepsArray:
    """Create a tensor for a tensorial type.

    This is the main entry point for turning a raw value into an
    :class:`e3nn_jax.IrrepsArray` using a "tensorial" descriptor.  Depending
    on the type of ``tensorial``, the value is either:

    * concatenated across the attributes of an :class:`IrrepsObj` subclass, or
    * concatenated across the entries of a mapping/frozen dict, or
    * wrapped directly into an :class:`e3nn_jax.IrrepsArray` if it is an
      :class:`Attr`, an :class:`e3nn_jax.Irreps`, or an irreps string.

    Args:
        tensorial: the descriptor describing the desired tensor structure.
        value: the raw value, of a type compatible with ``tensorial`` (typically
            a ``dict`` for structured types, an array for leaves).

    Returns:
        the resulting :class:`e3nn_jax.IrrepsArray`.

    Raises:
        TypeError: if ``tensorial`` is not a recognised tensorial type or a
            registered single-dispatch branch is not available.
    """
    try:
        # issubclass can fail if the value is not a class, so we guard against that here
        # and raise later with a more meaningful message
        is_subclass = issubclass(tensorial, IrrepsObj)
    except TypeError:
        pass  # Will raise at bottom of function
    else:
        if is_subclass:
            return create_tensor(tensorial_attrs(tensorial), value)

    raise TypeError(f"Unrecognised tensorial type: {tensorial.__class__.__name__}")


@create_tensor.register
def _create_tensor_irreps_obj(tensorial: IrrepsObj, value) -> e3j.IrrepsArray:
    """IrrepsObject leaf: delegate to its attribute mapping"""
    return create_tensor(tensorial_attrs(tensorial), value)


@create_tensor.register
def _create_tensor_dict(tensorial: dict, value) -> e3j.IrrepsArray:
    """Dict leaf: concatenate the per-key tensors in mapping order"""
    return e3j.concatenate(
        [create_tensor(attr, value[key]) for key, attr in tensorial.items()],
    )


@create_tensor.register
def _create_tensor_frozen_dict(tensorial: linen.FrozenDict, value):
    """FrozenDict leaf: unfreeze and delegate to the dict branch"""
    return create_tensor(tensorial.unfreeze(), value)


@create_tensor.register
def _create_tensor_irreps(  # pylint: disable=redefined-outer-name
    irreps: e3j.Irreps, value: Array
) -> e3j.IrrepsArray:
    """Irreps leaf: wrap the value in an IrrepsArray"""
    return e3j.IrrepsArray(irreps, value)


@create_tensor.register
def _create_tensor_str(  # pylint: disable=redefined-outer-name
    irreps: str, value: Array
) -> e3j.IrrepsArray:
    """String leaf: build the irreps from the string and wrap the value"""
    return e3j.IrrepsArray(irreps, value)


@create_tensor.register
def _create_tensor_attr(attr: Attr, value) -> e3j.IrrepsArray:
    """Attr leaf: delegate to the attribute's own create_tensor"""
    return attr.create_tensor(value)


@functools.singledispatch
def from_tensor(tensorial: Tensorial, value) -> ValueType:
    """Inverse of :func:`create_tensor`: split a tensor back into its parts.

    For an :class:`IrrepsObj` subclass or a mapping/frozen dict the incoming
    tensor is split into per-attribute tensors (in order of declaration) and
    each piece is passed to the corresponding leaf branch.  For leaf
    descriptors, the value is either validated (pure :class:`e3nn_jax.Irreps`)
    or returned through :meth:`Attr.from_tensor`.

    Args:
        tensorial: the descriptor describing the tensor's structure.
        value: the tensor to split.

    Returns:
        a value of a type compatible with ``tensorial`` (a mapping for
        structured types, the (validated) tensor for leaves).

    Raises:
        TypeError: if ``tensorial`` is not a recognised tensorial type.
        ValueError: if a leaf's irreps do not match the incoming tensor's
            irreps.
    """
    try:
        # issubclass can fail if the value is a class, so we guard against that here
        # and raise later with a more meaningful message
        is_subclass = issubclass(tensorial, IrrepsObj)
    except TypeError:
        pass  # Will raise at bottom of function
    else:
        if is_subclass:
            return from_tensor(tensorial_attrs(tensorial), value)

    raise TypeError(f"Unrecognised tensorial type: {tensorial.__class__.__name__}")


@from_tensor.register
def _from_tensor_irreps_obj(tensorial: IrrepsObj, value) -> dict[str, ValueType]:
    """IrrepsObject leaf: delegate to its attribute mapping"""
    return from_tensor(tensorial_attrs(tensorial), value)


@from_tensor.register
def _from_tensor_dict(tensorial: dict, value: Array) -> dict[str, ValueType]:
    """Dict leaf: split the tensor in attribute order and delegate per key."""
    dims = jnp.array(tuple(map(lambda val: irreps(val).dim, tensorial.values())))
    split_points = jnp.array(tuple(jnp.sum(dims[:i]) for i in range(len(dims) - 1)))
    split_value = jnp.split(value, split_points)

    return {
        key: from_tensor(dict_value, array_value)
        for array_value, (key, dict_value) in zip(split_value, tensorial_attrs(tensorial).items())
    }


@from_tensor.register
def _from_tensor_frozen_dict(tensorial: linen.FrozenDict, value):
    """FrozenDict leaf: unfreeze and delegate to the dict branch."""
    return from_tensor(tensorial.unfreeze(), value)


@from_tensor.register
def _from_tensor_irreps(  # pylint: disable=redefined-outer-name
    irreps: e3j.Irreps, value: e3j.IrrepsArray
) -> e3j.IrrepsArray:
    """Irreps leaf: check the incoming tensor has matching irreps."""
    if not irreps == value.irreps:
        raise ValueError(f"Irreps mismatch: {irreps} != {value.irreps}")
    return value


@from_tensor.register
def _from_tensor(attr: Attr, value) -> e3j.IrrepsArray:
    """Attr leaf: delegate to the attribute's own from_tensor."""
    return attr.from_tensor(value)


@functools.singledispatch
def tensorial_attrs(irreps_obj) -> dict[str, Tensorial]:
    """Return the tensorial attributes of an :class:`IrrepsObj` class.

    This is the single-dispatch entry point that maps an :class:`IrrepsObj`
    subclass (or a mapping / frozen dict) to an ordered mapping of its
    attribute names to their individual tensorial descriptors.

    The default (class) branch simply walks the class's attributes and picks
    out the ones that look like tensorial descriptors (i.e. not private and
    not callables).  Instance, dict and frozen-dict branches are registered
    below.

    Args:
        irreps_obj: an :class:`IrrepsObj` subclass (typical case) or any other
            supported tensorial descriptor.

    Returns:
        an ordered mapping of attribute name to tensorial descriptor.

    Raises:
        TypeError: if ``irreps_obj`` is not an :class:`IrrepsObj` subclass and
            no registered branch applies.
    """
    if issubclass(irreps_obj, IrrepsObj):
        return {
            name: val
            for name, val in vars(irreps_obj).items()
            if not (name.startswith("_") or callable(val))
        }

    raise TypeError(irreps_obj.__class__.__name__)


@tensorial_attrs.register
def _tensorial_attrs_irreps_obj(irreps_obj: IrrepsObj) -> dict[str, Tensorial]:
    """Instance branch: start from the class attributes and overlay instance-level ones."""
    attrs = tensorial_attrs(type(irreps_obj))
    attrs.update(
        {
            name: val
            for name, val in vars(irreps_obj).items()
            if not (name.startswith("_") or callable(val))
        }
    )
    return attrs


@tensorial_attrs.register
def _tensorial_attrs_dict(irreps_obj: dict) -> dict[str, Tensorial]:
    """Dict branch: return entries whose keys are not private."""
    return {name: val for name, val in irreps_obj.items() if not name.startswith("_")}


@tensorial_attrs.register
def _tensorial_attrs_frozen_dict(irreps_obj: linen.FrozenDict) -> dict[str, Tensorial]:
    """FrozenDict branch: unfreeze and delegate to the dict branch."""
    return tensorial_attrs(irreps_obj.unfreeze())


def get(irreps_obj: type[IrrepsObj], tensor: Array, attr_name: str = None) -> Array:
    """Extract a named attribute's slice from a flattened tensor.

    Given a tensor produced by :func:`create_tensor` for some
    :class:`IrrepsObj` type, this returns the contiguous slice of the tensor
    that corresponds to the attribute with the given name.

    Args:
        irreps_obj: the :class:`IrrepsObj` subclass (or compatible descriptor)
            that was used when the tensor was created.
        tensor: the flattened tensor (typically an :class:`e3nn_jax.IrrepsArray`
            or raw array) to slice.
        attr_name: name of the attribute whose slice to extract.  If ``None``
            or empty, ``tensor`` is returned unchanged.

    Returns:
        the slice of ``tensor`` corresponding to ``attr_name`` (or ``tensor``
        itself if ``attr_name`` is not given).

    Raises:
        ValueError: if ``attr_name`` is not present among the descriptor's
            attributes.
    """
    if not attr_name:
        return tensor

    attrs = tensorial_attrs(irreps_obj)
    idx = list(attrs.keys()).index(attr_name)

    # Get the linear start and end index of the tensor corresponding to the passed attribute
    begin = sum(irreps(attr).dim for attr in list(attrs.values())[:idx])
    end = begin + irreps(attrs[attr_name]).dim
    return tensor[begin:end]
