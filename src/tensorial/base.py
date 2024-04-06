# -*- coding: utf-8 -*-
import functools
from typing import Any, Dict, List, Mapping, Type, Union

import e3nn_jax as e3j
import equinox
from flax import linen
import jax
import jax.numpy as jnp
import numpy as np

__all__ = 'IrrepsObj', 'IrrepsTree', 'Attr', 'create', 'create_tensor', 'irreps', 'get', \
    'Tensorial', 'tensorial_attrs', 'from_tensor', 'as_array'

Array = Union[np.array, jax.Array]


def atleast_1d(arr) -> jnp.array:
    arr = jnp.asarray(arr)
    return arr if jnp.ndim(arr) >= 1 else jnp.reshape(arr, -1)


def as_array(arr: Array) -> jax.Array:
    if isinstance(arr, jax.Array):
        return arr

    return arr.array


class Attr(equinox.Module):
    """Irreps object attribute"""
    irreps: e3j.Irreps

    def __init__(self, irreps) -> None:  # pylint: disable=redefined-outer-name
        self.irreps = e3j.Irreps(irreps)

    def create_tensor(self, value: Any) -> e3j.IrrepsArray:
        return e3j.IrrepsArray(self.irreps, atleast_1d(value))

    def from_tensor(self, tensor: e3j.IrrepsArray) -> Any:
        """This can be overwritten to perform the backward transform of `create_tensor`"""
        return tensor


class IrrepsObj:
    """An object that contains tensorial attributes."""


IrrepsTree = Union[IrrepsObj, Dict]
Tensorial = Union[Attr, IrrepsObj, type(IrrepsObj), dict, linen.FrozenDict, e3j.Irreps]
ValueType = Union[Any, List['ValueType'], Dict[str, 'ValueType']]


@functools.singledispatch
def create(tensorial: Tensorial, value: Mapping):
    if not issubclass(tensorial, IrrepsObj):
        raise TypeError(tensorial.__class__.__name__)

    value_dict = {}
    for name, val in tensorial_attrs(tensorial).items():
        value_dict[name] = create(val, value[name])

    return value_dict


@create.register
def _(attr: Attr, value) -> e3j.IrrepsArray:
    """Leaf, so create the tensor"""
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
    """Get the irreps for a tensorial type"""
    if not issubclass(tensorial, IrrepsObj):
        raise TypeError(tensorial.__class__.__name__)

    # IrrepsObj code:
    total_irreps = None

    for name, val in tensorial_attrs(tensorial).items():
        try:
            total_irreps = val.irreps if total_irreps is None else total_irreps + val.irreps
        except AttributeError as exc:
            raise AttributeError(f'Failed to get irreps for {name}') from exc

    return total_irreps


@irreps.register
def _irreps(attr: Attr) -> e3j.Irreps:
    return attr.irreps


@irreps.register
def _irreps(tensorial: e3j.Irreps) -> e3j.Irreps:
    return tensorial


@functools.singledispatch
def create_tensor(tensorial: Tensorial, value: ValueType) -> e3j.IrrepsArray:
    """Create a tensor for a tensorial type"""
    try:
        # issubclass can fail if the value is not a class, so we guard against that here
        # and raise later with a more meaningful message
        is_subclass = issubclass(tensorial, IrrepsObj)
    except TypeError as _exc:
        pass  # Will raise at bottom of function
    else:
        if is_subclass:
            return create_tensor(tensorial_attrs(tensorial), value)

    raise TypeError(f'Unrecognised tensorial type: {tensorial.__class__.__name__}')


@create_tensor.register
def _create_tensor(tensorial: IrrepsObj, value) -> e3j.IrrepsArray:
    return create_tensor(tensorial_attrs(tensorial), value)


@create_tensor.register
def _create_tensor(tensorial: dict, value) -> e3j.IrrepsArray:
    return e3j.concatenate([create_tensor(attr, value[key]) for key, attr in tensorial.items()],)


@create_tensor.register
def _create_tensor(tensorial: linen.FrozenDict, value):
    return create_tensor(tensorial.unfreeze(), value)


@create_tensor.register
def _create_tensor(irreps: e3j.Irreps, value: Array) -> e3j.IrrepsArray:  # pylint: disable=redefined-outer-name
    return e3j.IrrepsArray(irreps, value)


@create_tensor.register
def _create_tensor(attr: Attr, value) -> e3j.IrrepsArray:
    return attr.create_tensor(value)


@functools.singledispatch
def from_tensor(tensorial: Tensorial, value) -> ValueType:
    """Create a tensor for a tensorial type"""
    try:
        # issubclass can fail if the value is a class, so we guard against that here
        # and raise later with a more meaningful message
        is_subclass = issubclass(tensorial, IrrepsObj)
    except TypeError as _exc:
        pass  # Will raise at bottom of function
    else:
        if is_subclass:
            return from_tensor(tensorial_attrs(tensorial), value)

    raise TypeError(f'Unrecognised tensorial type: {tensorial.__class__.__name__}')


@from_tensor.register
def _from_tensor(tensorial: IrrepsObj, value) -> Dict[str, ValueType]:
    return from_tensor(tensorial_attrs(tensorial), value)


@from_tensor.register
def _from_tensor(tensorial: dict, value: Array) -> Dict[str, ValueType]:
    dims = jnp.array(tuple(map(lambda val: irreps(val).dim, tensorial.values())))
    split_points = jnp.array(tuple(jnp.sum(dims[:i]) for i in range(len(dims) - 1)))
    split_value = jnp.split(value, split_points)

    return {
        key: from_tensor(dict_value, array_value)
        for array_value, (key, dict_value) in zip(split_value,
                                                  tensorial_attrs(tensorial).items())
    }


@from_tensor.register
def _from_tensor(tensorial: linen.FrozenDict, value):
    return from_tensor(tensorial.unfreeze(), value)


@from_tensor.register
def _from_tensor(irreps: e3j.Irreps, value: e3j.IrrepsArray) -> e3j.IrrepsArray:  # pylint: disable=redefined-outer-name
    # Nothing to do
    if not irreps == value.irreps:
        raise ValueError(f'Irreps mismatch: {irreps} != {value.irreps}')
    return value


@from_tensor.register
def _from_tensor(attr: Attr, value) -> e3j.IrrepsArray:
    return attr.from_tensor(value)


@functools.singledispatch
def tensorial_attrs(irreps_obj) -> Dict[str, Tensorial]:
    if issubclass(irreps_obj, IrrepsObj):
        return {name: val for name, val in vars(irreps_obj).items() if not (name.startswith('_') or callable(val))}

    raise TypeError(irreps_obj.__class__.__name__)


@tensorial_attrs.register
def _tensorial_attrs(irreps_obj: IrrepsObj) -> Dict[str, Tensorial]:
    """Get the irrep attributes for the passed object"""
    attrs = tensorial_attrs(type(irreps_obj))
    attrs.update({name: val for name, val in vars(irreps_obj).items() if not (name.startswith('_') or callable(val))})
    return attrs


@tensorial_attrs.register
def _tensorial_attrs(irreps_obj: dict) -> Dict[str, Tensorial]:
    return {name: val for name, val in irreps_obj.items() if not name.startswith('_')}


@tensorial_attrs.register
def _tensorial_attrs(irreps_obj: linen.FrozenDict) -> Dict[str, Tensorial]:
    return tensorial_attrs(irreps_obj.unfreeze())


def get(irreps_obj: Type[IrrepsObj], tensor: Array, attr_name: str = None) -> Array:
    if not attr_name:
        return tensor

    attrs = tensorial_attrs(irreps_obj)
    idx = list(attrs.keys()).index(attr_name)

    # Get the linear start and end index of the tensor corresponding to the passed attribute
    begin = sum(irreps(attr).dim for attr in list(attrs.values())[:idx])
    end = begin + irreps(attrs[attr_name]).dim
    return tensor[begin:end]
