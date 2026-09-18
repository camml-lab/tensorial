import e3nn_jax as e3j
import jax
import jax.numpy as jnp
import jraph
import numpy as np

from tensorial import gcnn
from tensorial.gcnn import mace

from ... import utils


def test_mace(cube_graph: jraph.GraphsTuple):
    r_max = 5.0
    num_types = 3

    model = utils.graph_model(
        r_max,
        e3j.Irreps("0e + 1o + 2e"),
        mace.Mace(
            irreps_out=e3j.Irreps("0e"),
            out_field=gcnn.atomic.ENERGY_PER_ATOM,
            hidden_irreps="2x0e + 2x1o",
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


def test_mace_dict_avg_num_neighbours(cube_graph: jraph.GraphsTuple):
    r_max = 5.0
    num_types = 3
    avg_num_neighbours = {0: 1.0, 1: 2.0, 2: 3.0}

    model = utils.graph_model(
        r_max,
        e3j.Irreps("0e + 1o + 2e"),
        mace.Mace(
            irreps_out=e3j.Irreps("0e"),
            out_field=gcnn.atomic.ENERGY_PER_ATOM,
            hidden_irreps="2x0e + 2x1o",
            num_types=num_types,
            y0_values=np.random.rand(num_types).tolist(),
            avg_num_neighbours=avg_num_neighbours,
        ),
        type_numbers=[0, 1, 2],
    )

    params = model.init(jax.random.PRNGKey(0), cube_graph)

    # Check that apply doesn't crash
    model.apply(params, cube_graph)

    # Check for equivariance
    def wrapper(positions: e3j.IrrepsArray) -> e3j.IrrepsArray:
        # Need to create a new graph to avoid mutating the original
        new_nodes = cube_graph.nodes.copy()
        new_nodes[gcnn.keys.POSITIONS] = positions.array
        new_graph = cube_graph._replace(nodes=new_nodes)
        outs = model.apply(params, new_graph)
        return e3j.as_irreps_array(e3j.sum(outs.nodes[gcnn.atomic.ENERGY_PER_ATOM], axis=1))

    e3j.utils.assert_equivariant(
        wrapper,
        jax.random.PRNGKey(1),
        e3j.IrrepsArray("1o", cube_graph.nodes[gcnn.keys.POSITIONS]),
    )


def test_interaction_block_normalization():
    # Test InteractionBlock directly
    avg_num_neighbours = {0: 1.0, 1: 4.0}  # Square roots will be 1.0 and 2.0

    # Use a simple block
    block = mace.InteractionBlock("1x0e", avg_num_neighbours=avg_num_neighbours)

    # Create dummy graph data
    n_node = 2
    n_edge = 1
    node_features = e3j.IrrepsArray("0e", jnp.ones((n_node, 1)))
    edge_features = e3j.IrrepsArray("0e", jnp.ones((n_edge, 1)))
    radial_embedding = jnp.ones((n_edge, 1))
    senders = jnp.array([0])
    receivers = jnp.array([1])
    node_types = jnp.array([0, 1])

    # Initialize block
    params = block.init(
        jax.random.PRNGKey(0),
        node_features,
        edge_features,
        radial_embedding,
        senders,
        receivers,
        node_types=node_types,
    )

    block.apply(
        params,
        node_features,
        edge_features,
        radial_embedding,
        senders,
        receivers,
        node_types=node_types,
    )
