# -*- coding: utf-8 -*-
from typing import Dict, List, Optional, Tuple, Union

import e3nn_jax as e3j
import jax
import jax.numpy as jnp

from . import base

__all__ = 'SphericalHarmonic', 'CartesianTensor', 'OneHot'


class SphericalHarmonic(base.Attr):
    """An attribute that is the spherical harmonics evaluated as some values"""
    normalize: bool
    normalization: Optional[str] = None
    algorithm: Optional[Tuple[str]] = None

    def __init__(
        self,
        irreps,
        normalize,
        normalization: str = None,
        *,
        algorithm: Tuple[str] = None,
    ):
        super().__init__(irreps)
        self.normalize = normalize
        self.normalization = normalization
        self.algorithm = algorithm

    def create_tensor(self, value) -> jnp.array:
        return e3j.spherical_harmonics(
            self.irreps,
            value,
            normalize=self.normalize,
            normalization=self.normalization,
            algorithm=self.algorithm,
        )


class OneHot(base.Attr):
    """One-hot encoding as a direct sum of even scalars"""

    def __init__(self, num_classes):
        super().__init__(num_classes * e3j.Irrep(0, 1))

    @property
    def num_classes(self) -> int:
        return self.irreps[0].mul

    def create_tensor(self, value) -> jnp.array:
        return e3j.IrrepsArray(self.irreps, jax.nn.one_hot(value, self.num_classes))


class CartesianTensor(base.Attr):
    formula: str
    keep_ir: Optional[Union[e3j.Irreps, List[e3j.Irrep]]]
    irreps_dict: Dict
    change_of_basis: jax.Array
    _indices: str

    def __init__(self, formula: str, keep_ir=None, **irreps_dict) -> None:
        self.formula = formula
        self.keep_ir = keep_ir
        self.irreps_dict = irreps_dict
        self._indices = formula.split('=')[0].replace('-', '')

        # Construct the change of basis arrays
        cob = e3j.reduced_tensor_product_basis(formula, keep_ir=self.keep_ir, **self.irreps_dict)
        self.change_of_basis = cob.array
        super().__init__(cob.irreps)

    def create_tensor(self, value) -> e3j.IrrepsArray:
        return super().create_tensor(jnp.einsum('ij,ijz->z', value, self.change_of_basis))

    def from_tensor(self, tensor: e3j.IrrepsArray) -> jax.Array:
        """
        Take an irrep tensor and perform the change of basis transformation back to a Cartesian tensor
        :param tensor: the irrep tensor
        :return: the Cartesian tensor
        """
        rot = self.change_of_basis.reshape(-1, self.change_of_basis.shape[-1])
        cartesian = base.as_array(tensor) @ rot.T
        return cartesian.reshape((*tensor.shape[:-1], *self.change_of_basis.shape[:-1]))
