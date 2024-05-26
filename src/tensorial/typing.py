# -*- coding: utf-8 -*-
from typing import Tuple, Union

import e3nn_jax as e3j

IrrepLike = Union[str, e3j.Irrep]
IrrepsLike = Union[str, e3j.Irreps, Tuple[e3j.MulIrrep]]
