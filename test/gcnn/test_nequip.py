# -*- coding: utf-8 -*-
import e3nn_jax as e3j
import jraph

from tensorial import gcnn
from tensorial.gcnn import _nequip, keys


def test_nequip_interaction_block(cube_graph_gcnn: jraph.GraphsTuple, rng_key):
    irreps_out = e3j.Irreps("0e + 1o + 2e")

    block = _nequip.InteractionBlock(irreps_out)

    args = (
        cube_graph_gcnn.nodes[keys.FEATURES],
        cube_graph_gcnn.edges[keys.ATTRIBUTES],
        cube_graph_gcnn.edges[keys.RADIAL_EMBEDDINGS],
        cube_graph_gcnn.senders,
        cube_graph_gcnn.receivers,
    )
    params = block.init(rng_key, *args)
    node_features = block.apply(params, *args)

    assert isinstance(node_features, e3j.IrrepsArray)
    assert node_features.irreps == irreps_out


def test_nequip_layer(cube_graph_gcnn: jraph.GraphsTuple, rng_key):
    irreps_out = e3j.Irreps("0e + 1o + 2e")

    layer = gcnn.NequipLayer(irreps_out)

    params = layer.init(rng_key, cube_graph_gcnn)
    graph_out = layer.apply(params, cube_graph_gcnn)

    assert graph_out.nodes[keys.FEATURES].irreps == irreps_out

    def wrapper(positions):
        cube_graph_gcnn.nodes[keys.POSITIONS] = positions
        out = layer.apply(params, cube_graph_gcnn)
        return out.nodes[keys.POSITIONS]

    e3j.utils.assert_equivariant(wrapper, rng_key, cube_graph_gcnn.nodes[keys.POSITIONS])
