from typing import Final

import e3nn_jax as e3j
import jax
import jax.numpy as jnp
import jraph

from tensorial import nn_utils
from tensorial.gcnn import _message_passing, keys


def test_message_passing(cube_graph_gcnn: jraph.GraphsTuple, rng_key):
    irreps_out = e3j.Irreps("0e + 1o + 2e")

    conv = _message_passing.MessagePassingConvolution(irreps_out)

    args = (
        cube_graph_gcnn.nodes[keys.FEATURES],
        cube_graph_gcnn.edges[keys.ATTRIBUTES],
        cube_graph_gcnn.edges[keys.RADIAL_EMBEDDINGS],
        cube_graph_gcnn.senders,
        cube_graph_gcnn.receivers,
    )
    params = conv.init(rng_key, *args)
    node_features = conv.apply(params, *args)

    assert isinstance(node_features, e3j.IrrepsArray)
    assert node_features.irreps == irreps_out * cube_graph_gcnn.n_node[0].item()

    # Now, let's check normalization
    NORM_FACTOR: Final[float] = 42.0
    conv_with_normalization = _message_passing.MessagePassingConvolution(
        irreps_out, avg_num_neighbours=NORM_FACTOR
    )
    node_features_normalized = conv_with_normalization.apply(params, *args)

    assert jnp.allclose(node_features_normalized.array, node_features.array / jnp.sqrt(NORM_FACTOR))


def test_message_passing_normalize_by_type(cube_graph_gcnn: jraph.GraphsTuple, rng_key):
    NUM_TYPES: Final[int] = 3
    IRREPS_OUT: Final[e3j.Irreps] = e3j.Irreps("0e + 1o + 2e")

    rng_key, subkey = jax.random.split(rng_key)
    types = list(range(NUM_TYPES))
    avg_neighs = jax.random.uniform(subkey, (NUM_TYPES,)) + 10.0
    norms_dict = dict(zip(types, avg_neighs.tolist()))

    conv = _message_passing.MessagePassingConvolution(IRREPS_OUT)
    conv_with_normalization = _message_passing.MessagePassingConvolution(
        IRREPS_OUT, avg_num_neighbours=norms_dict
    )

    # Initialise the module
    args = (
        cube_graph_gcnn.nodes[keys.FEATURES],
        cube_graph_gcnn.edges[keys.ATTRIBUTES],
        cube_graph_gcnn.edges[keys.RADIAL_EMBEDDINGS],
        cube_graph_gcnn.senders,
        cube_graph_gcnn.receivers,
    )
    node_types = jax.random.randint(
        rng_key, (cube_graph_gcnn.n_node[0].item(),), minval=0, maxval=NUM_TYPES
    )
    params = conv.init(rng_key, *args)
    params_norm = conv_with_normalization.init(rng_key, *args, node_types=node_types)

    # Calculate the two versions
    node_features = conv.apply(params, *args)
    node_features_normalized = conv_with_normalization.apply(
        params_norm, *args, node_types=node_types
    )
    norm_values = avg_neighs[nn_utils.vwhere(node_types, jnp.asarray(types))].reshape(-1, 1)

    # Check that normalisation was correctly applied
    assert jnp.allclose(node_features_normalized.array, node_features.array / jnp.sqrt(norm_values))
