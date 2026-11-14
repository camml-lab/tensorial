import e3nn_jax as e3j
import jax
import jraph
import numpy as np
import pytest

import tensorial
from tensorial import gcnn
from tensorial.gcnn import _message_passing, keys, nequip
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


def test_nequip_resnet_skipped_when_irreps_change(cube_graph_gcnn: jraph.GraphsTuple, rng_key):
    """The input features (one-hot scalars) differ from the hidden irreps, so the first layer
    must skip the residual, while later layers (same irreps in and out) apply it."""
    # Every irrep is reachable from the edge spherical harmonics (1o + 2e) in a single layer, so
    # layer 1 maps 8x0e -> hidden_irreps and layer 2 maps hidden_irreps -> hidden_irreps
    hidden_irreps = e3j.Irreps("4x0e + 4x1o + 4x2e")
    assert cube_graph_gcnn.nodes[keys.FEATURES].irreps != hidden_irreps

    layer_kwargs = dict(num_species=3)
    params = gcnn.Nequip(hidden_irreps, num_layers=2, **layer_kwargs).init(rng_key, cube_graph_gcnn)
    params_0 = {"params": params["params"]["_layers_0"]}
    params_1 = {"params": params["params"]["_layers_1"]}

    def features(graph):
        return graph.nodes[keys.FEATURES]

    # Layer 1: irreps change, so resnet=True must be a no-op (and must not raise)
    plain_0 = gcnn.NequipLayer(hidden_irreps, **layer_kwargs).apply(params_0, cube_graph_gcnn)
    resnet_0 = gcnn.NequipLayer(hidden_irreps, resnet=True, **layer_kwargs).apply(
        params_0, cube_graph_gcnn
    )
    assert features(resnet_0).irreps == hidden_irreps
    np.testing.assert_allclose(features(resnet_0).array, features(plain_0).array)

    # Layer 2: irreps unchanged, so resnet=True adds the input features back
    plain_1 = gcnn.NequipLayer(hidden_irreps, **layer_kwargs).apply(params_1, plain_0)
    resnet_1 = gcnn.NequipLayer(hidden_irreps, resnet=True, **layer_kwargs).apply(params_1, plain_0)
    np.testing.assert_allclose(
        features(resnet_1).array,
        features(plain_1).array + features(plain_0).array,
        rtol=1e-6,
        atol=1e-6,
    )

    # The full stack runs with resnet=True and matches the layer-by-layer result
    stack = gcnn.Nequip(hidden_irreps, num_layers=2, resnet=True, **layer_kwargs)
    np.testing.assert_allclose(
        features(stack.apply(params, cube_graph_gcnn)).array,
        features(resnet_1).array,
        rtol=1e-6,
        atol=1e-6,
    )


def test_nequip_per_type_avg_num_neighbours(cube_graph_gcnn: jraph.GraphsTuple, rng_key):
    """A per-type `avg_num_neighbours` is looked up using the node species."""
    irreps_out = e3j.Irreps("0e + 1o + 2e")
    args = (
        cube_graph_gcnn.nodes[keys.FEATURES],
        cube_graph_gcnn.edges[keys.ATTRIBUTES],
        cube_graph_gcnn.edges[keys.RADIAL_EMBEDDINGS],
        cube_graph_gcnn.senders,
        cube_graph_gcnn.receivers,
    )
    node_species = cube_graph_gcnn.nodes[keys.SPECIES][:, 0]

    def compute(avg_num_neighbours):
        block = nequip.InteractionBlock(
            irreps_out, num_species=3, avg_num_neighbours=avg_num_neighbours
        )
        params = block.init(rng_key, *args, node_species=node_species)
        return block.apply(params, *args, node_species=node_species).array

    # The same value for every type is equivalent to a single global value
    np.testing.assert_allclose(
        compute({0: 4.0, 1: 4.0, 2: 4.0}), compute(4.0), rtol=1e-6, atol=1e-6
    )

    # Exact check on the aggregated messages: each node is divided by its own type's value
    avg = {0: 1.0, 1: 4.0, 2: 9.0}
    msg_args = (
        cube_graph_gcnn.nodes[keys.FEATURES],
        *args[1:],
    )
    per_type = _message_passing.MessagePassingConvolution(irreps_out, avg_num_neighbours=avg)
    glob = _message_passing.MessagePassingConvolution(irreps_out, avg_num_neighbours=1.0)
    params = glob.init(rng_key, *msg_args)
    expected = glob.apply(params, *msg_args).array
    result = per_type.apply(params, *msg_args, node_types=node_species).array
    scale = np.sqrt(np.array([avg[int(t)] for t in node_species]))[:, None]
    np.testing.assert_allclose(result * scale, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("skip_connection_first_layer", [True, False])
def test_nequip_skip_connection_first_layer(
    cube_graph_gcnn: jraph.GraphsTuple, rng_key, skip_connection_first_layer
):
    model = gcnn.Nequip(
        e3j.Irreps("4x0e + 4x1o + 4x2e"),
        num_layers=2,
        num_species=3,
        skip_connection_first_layer=skip_connection_first_layer,
    )
    params = model.init(rng_key, cube_graph_gcnn)

    def has_skip(layer: str) -> bool:
        return "skip_connection" in params["params"][layer]["_interaction_block"]

    assert has_skip("_layers_0") == skip_connection_first_layer
    assert has_skip("_layers_1")

    # With every edge masked out, each atom is isolated.  Without a first-layer self-connection
    # nothing can make its features non-zero, so the output must vanish
    edges = dict(cube_graph_gcnn.edges)
    edges[keys.MASK] = np.zeros(len(cube_graph_gcnn.senders), dtype=bool)
    isolated = cube_graph_gcnn._replace(edges=edges)
    out = model.apply(params, isolated).nodes[keys.FEATURES].array
    if skip_connection_first_layer:
        assert not np.allclose(out, 0.0)
    else:
        np.testing.assert_allclose(out, 0.0, atol=1e-7)
