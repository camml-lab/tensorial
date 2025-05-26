import e3nn_jax as e3j
import jax
import jraph
import numpy as np
import pytest

from tensorial import gcnn
from tensorial.gcnn import _nequip, keys


def test_nequip_interaction_block(cube_graph_gcnn: jraph.GraphsTuple, rng_key):
    irreps_out = e3j.Irreps("0e + 1o + 2e")
    block = _nequip.InteractionBlock(irreps_out, num_species=3)

    args = (
        cube_graph_gcnn.nodes[keys.FEATURES],
        cube_graph_gcnn.edges[keys.ATTRIBUTES],
        cube_graph_gcnn.edges[keys.RADIAL_EMBEDDINGS],
        cube_graph_gcnn.senders,
        cube_graph_gcnn.receivers,
        cube_graph_gcnn.nodes[keys.SPECIES],
    )
    params = block.init(rng_key, *args)
    node_features = block.apply(params, *args)

    assert isinstance(node_features, e3j.IrrepsArray)
    assert node_features.irreps == irreps_out


@pytest.mark.parametrize("skip_connection", [True, False])
def test_nequip_interaction_block_with_padding(
    cube_graph_gcnn: jraph.GraphsTuple, rng_key, skip_connection
):
    irreps_out = e3j.Irreps("0e + 1o + 2e")
    block = _nequip.InteractionBlock(irreps_out, num_species=3, skip_connection=skip_connection)

    def _compute(graph: jraph.GraphsTuple):
        args = (
            graph.nodes[keys.FEATURES],
            graph.edges[keys.ATTRIBUTES],
            graph.edges[keys.RADIAL_EMBEDDINGS],
            graph.senders,
            graph.receivers,
        )
        kwargs = {"node_species": graph.nodes[keys.SPECIES]}
        if "mask" in graph.nodes:
            kwargs["node_mask"] = graph.nodes["mask"]
        if "mask" in graph.edges:
            kwargs["edge_mask"] = graph.edges["mask"]
        params = block.init(rng_key, *args, **kwargs)
        return block.apply(params, *args, **kwargs).array.sum()

    without_padding = _compute(cube_graph_gcnn)

    padded = gcnn.data.pad_with_graphs(
        cube_graph_gcnn, cube_graph_gcnn.n_node[0] + 1, cube_graph_gcnn.n_edge[0], 2
    )
    with_padding = _compute(padded)
    assert np.isclose(without_padding, with_padding)


def test_nequip_layer(cube_graph_gcnn: jraph.GraphsTuple, rng_key):
    irreps_out = e3j.Irreps("0e + 1o + 2e")

    layer = gcnn.NequipLayer(irreps_out)

    params = layer.init(rng_key, cube_graph_gcnn)
    graph_out = layer.apply(params, cube_graph_gcnn)

    assert graph_out.nodes[keys.FEATURES].irreps == irreps_out

    def wrapper(positions: e3j.IrrepsArray) -> e3j.IrrepsArray:
        cube_graph_gcnn.nodes[keys.POSITIONS] = positions.array
        out = layer.apply(params, cube_graph_gcnn)
        return e3j.IrrepsArray("1o", out.nodes[keys.POSITIONS])

    e3j.utils.assert_equivariant(
        wrapper, rng_key, e3j.IrrepsArray("1o", cube_graph_gcnn.nodes[keys.POSITIONS])
    )
