# -*- coding: utf-8 -*-
from typing import Optional

import e3nn_jax as e3j
from e3nn_jax import legacy
from flax import linen
import jax.numpy as jnp


class FlaxFullyConnectedTensorProduct(linen.Module):
    """Flax version of an equivariant, fully connected tensor product

    Follow implementation of haiku version in e3nn_jax:
        https://github.com/e3nn/e3nn-jax/blob/main/e3nn_jax/_src/fc_tp_haiku.py
    """

    irreps_out: e3j.Irreps
    irreps_in1: Optional[e3j.Irreps] = None
    irreps_in2: Optional[e3j.Irreps] = None

    @linen.compact
    def __call__(self, x1: e3j.IrrepsArray, x2: e3j.IrrepsArray, **kwargs) -> e3j.IrrepsArray:
        irreps_out = e3j.Irreps(self.irreps_out)
        irreps_in1 = e3j.Irreps(self.irreps_in1) if self.irreps_in1 is not None else None
        irreps_in2 = e3j.Irreps(self.irreps_in2) if self.irreps_in2 is not None else None

        x1 = e3j.as_irreps_array(x1)
        x2 = e3j.as_irreps_array(x2)

        leading_shape = jnp.broadcast_shapes(x1.shape[:-1], x2.shape[:-1])
        x1 = x1.broadcast_to(leading_shape + (-1,))
        x2 = x2.broadcast_to(leading_shape + (-1,))

        if irreps_in1 is not None:
            x1 = x1.rechunk(irreps_in1)
        if irreps_in2 is not None:
            x2 = x2.rechunk(irreps_in2)

        x1 = x1.remove_zero_chunks().simplify()
        x2 = x2.remove_zero_chunks().simplify()

        tp = legacy.FunctionalFullyConnectedTensorProduct(x1.irreps, x2.irreps, irreps_out.simplify())
        ws = [
            self.param(
                (
                    f'w[{ins.i_in1},{ins.i_in2},{ins.i_out}] '
                    f'{tp.irreps_in1[ins.i_in1]},{tp.irreps_in2[ins.i_in2]},{tp.irreps_out[ins.i_out]}'
                ),
                linen.initializers.normal(stddev=ins.weight_std),
                ins.path_shape,
            ) for ins in tp.instructions
        ]
        f = lambda x1, x2: tp.left_right(ws, x1, x2, **kwargs)

        for _ in range(len(leading_shape)):
            f = e3j.utils.vmap(f)

        output = f(x1, x2)
        return output.rechunk(irreps_out)
