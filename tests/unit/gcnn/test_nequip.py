import e3nn_jax as e3j
import jax
import jraph
import numpy as np
import pytest

import tensorial
from tensorial import gcnn
from tensorial.gcnn import keys, nequip
import tensorial.nn


def test_nequip_interaction_block(cube_graph_gcnn: jraph.GraphsTuple, rng_key):
    irreps_out = e3j.Irreps("0e + 1o + 2e")
    block = nequip.InteractionBlock(irreps_out, num_species=3)

    args = (
        cube_graph_gcnn.nodes[keys.FEATURES],
        cube_graph_gcnn.edges[keys.ATTRIBUTES],
        cube_graph_gcnn.edges[keys.RADIAL_EMBEDDINGS],
        cube_graph_gcnn.senders,
        cube_graph_gcnn.receivers,
    )
    kwargs = {"node_species": cube_graph_gcnn.nodes[keys.SPECIES][:, 0]}

    params = block.init(rng_key, *args, **kwargs)
    node_features = block.apply(params, *args, **kwargs)

    assert isinstance(node_features, e3j.IrrepsArray)
    assert node_features.irreps == irreps_out


@pytest.mark.parametrize("skip_connection", [True, False])
def test_nequip_interaction_block_with_padding(
    cube_graph_gcnn: jraph.GraphsTuple, rng_key, skip_connection
):
    irreps_out = e3j.Irreps("0e + 1o + 2e")
    block = nequip.InteractionBlock(irreps_out, num_species=3, skip_connection=skip_connection)

    def _compute(graph: jraph.GraphsTuple):
        args = (
            graph.nodes[keys.FEATURES],
            graph.edges[keys.ATTRIBUTES],
            graph.edges[keys.RADIAL_EMBEDDINGS],
            graph.senders,
            graph.receivers,
        )
        kwargs = {"node_species": graph.nodes[keys.SPECIES][:, 0]}
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


def test_nequip_stack(cube_graph_gcnn: jraph.GraphsTuple, rng_key):
    hidden_irreps = tensorial.make_irreps(4, 2)
    model = gcnn.Nequip(hidden_irreps, num_layers=3, num_species=3)

    params = model.init(rng_key, cube_graph_gcnn)
    graph_out = model.apply(params, cube_graph_gcnn)

    assert sorted(params["params"]) == ["_layers_0", "_layers_1", "_layers_2"]
    # The gate returns the irreps in canonical (regrouped) order
    assert graph_out.nodes[keys.FEATURES].irreps == hidden_irreps.regroup()

    def wrapper(positions: e3j.IrrepsArray) -> e3j.IrrepsArray:
        cube_graph_gcnn.nodes[keys.POSITIONS] = positions.array
        out = model.apply(params, cube_graph_gcnn)
        return e3j.IrrepsArray("1o", out.nodes[keys.POSITIONS])

    e3j.utils.assert_equivariant(
        wrapper, rng_key, e3j.IrrepsArray("1o", cube_graph_gcnn.nodes[keys.POSITIONS])
    )


def test_nequip_matches_stacked_layers(cube_graph_gcnn: jraph.GraphsTuple, rng_key):
    """`Nequip` is exactly a chain of `NequipLayer`s: given the same weights, it produces the
    same node features as the equivalent hand-written stack."""
    hidden_irreps = tensorial.make_irreps(4, 2)
    layer_kwargs = dict(num_species=3, avg_num_neighbours=3.0)

    stacked = tensorial.nn.Sequential(
        [gcnn.NequipLayer(hidden_irreps, **layer_kwargs) for _ in range(2)]
    )
    model = gcnn.Nequip(hidden_irreps, num_layers=2, **layer_kwargs)

    stacked_params = stacked.init(rng_key, cube_graph_gcnn)
    params = {"params": {f"_layers_{i}": stacked_params["params"][f"layers_{i}"] for i in range(2)}}

    expected = stacked.apply(stacked_params, cube_graph_gcnn).nodes[keys.FEATURES]
    result = model.apply(params, cube_graph_gcnn).nodes[keys.FEATURES]

    assert result.irreps == expected.irreps
    np.testing.assert_allclose(result.array, expected.array, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("num_species", [1, 3])
def test_nequip_num_species_sets_self_connection_weights(
    cube_graph_gcnn: jraph.GraphsTuple, rng_key, num_species
):
    model = gcnn.Nequip(tensorial.make_irreps(2, 1), num_layers=1, num_species=num_species)
    params = model.init(rng_key, cube_graph_gcnn)

    skip = params["params"]["_layers_0"]["_interaction_block"]["skip_connection"]
    assert all(weights.shape[0] == num_species for weights in jax.tree_util.tree_leaves(skip))


def test_nequip_rejects_no_layers(cube_graph_gcnn: jraph.GraphsTuple, rng_key):
    with pytest.raises(ValueError, match="num_layers"):
        gcnn.Nequip(tensorial.make_irreps(2, 1), num_layers=0).init(rng_key, cube_graph_gcnn)
