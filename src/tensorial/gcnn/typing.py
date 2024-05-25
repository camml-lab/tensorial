# -*- coding: utf-8 -*-
from typing import TypeVar

import jax
import jaxtyping as jt

Mask = jt.Bool[jax.Array, "masked"]
IndexArray = jt.Int[jax.Array, "index"]
Shape = TypeVar("Shape")
