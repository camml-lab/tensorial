import functools

import e3nn_jax as e3j
import jax
import jraph
import numpy as np
import utils

from tensorial import gcnn
from tensorial.gcnn import _mace


def test_symmetric_contraction():
    num_types = 4
    x = e3j.normal("0e + 0o + 1o + 1e + 2e + 2o", jax.random.PRNGKey(0), (32, 128))
    types = jax.random.randint(jax.random.PRNGKey(1), (32,), minval=0, maxval=num_types)

    contraction = _mace.SymmetricContraction(3, ["0e", "1o", "2e"], num_types=num_types)
    params = contraction.init(jax.random.PRNGKey(0), x, types)

    e3j.utils.assert_equivariant(
        functools.partial(contraction.apply, params, input_type=types), jax.random.PRNGKey(3), x
    )


def test_mace(cube_graph: jraph.GraphsTuple):
    r_max = 5.0
    num_types = 3

    model = utils.graph_model(
        r_max,
        e3j.Irreps("0e + 1o + 2e"),
        _mace.Mace(
            irreps_out=e3j.Irreps("0e"),
            out_field=gcnn.atomic.ENERGY_PER_ATOM,
            hidden_irreps=" 2x0e + 2x1o",
            num_types=num_types,
            y0_values=np.random.rand(num_types).tolist(),
        ),
        type_numbers=[0],
    )

    params = model.init(jax.random.PRNGKey(0), cube_graph)

    def wrapper(positions: e3j.IrrepsArray) -> e3j.IrrepsArray:
        cube_graph.nodes[gcnn.keys.POSITIONS] = positions.array
        outs = model.apply(params, cube_graph)
        return e3j.as_irreps_array(e3j.sum(outs.nodes[gcnn.atomic.ENERGY_PER_ATOM], axis=1))

    e3j.utils.assert_equivariant(
        wrapper,
        jax.random.PRNGKey(1),
        e3j.IrrepsArray("1o", cube_graph.nodes[gcnn.keys.POSITIONS]),
    )
