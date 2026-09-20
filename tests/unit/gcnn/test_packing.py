import e3nn_jax as e3j
import jax
import jax.numpy as jnp
import jraph
import pytest

from tensorial import base, gcnn
from tensorial.gcnn import _packing, keys


def test_pack_nodes(cube_graph_gcnn: jraph.GraphsTuple, rng_key):
    # Pack nodes.features into nodes.attributes
    # cube_graph_gcnn.nodes[keys.FEATURES] is 8x0e (dim 8)
    packer = _packing.Pack(
        attrs={"nodes.features": base.Attr("8x0e")},
        out_field="nodes.attributes",
    )

    # Ensure it's a raw array to avoid IrrepsArray conversion issues
    graph_dict = cube_graph_gcnn._asdict()
    graph_dict["nodes"] = dict(graph_dict["nodes"])
    graph_dict["nodes"][keys.FEATURES] = graph_dict["nodes"][keys.FEATURES].array
    graph = jraph.GraphsTuple(**graph_dict)

    params = packer.init(rng_key, graph)
    graph_out = packer.apply(params, graph)

    assert "attributes" in graph_out.nodes
    assert isinstance(graph_out.nodes["attributes"], e3j.IrrepsArray)
    assert graph_out.nodes["attributes"].irreps == e3j.Irreps("8x0e")


def test_pack_broadcast_globals_to_nodes(cube_graph_gcnn: jraph.GraphsTuple, rng_key):
    # Pack globals.temperature into nodes.attributes

    graph = cube_graph_gcnn._asdict()
    graph["globals"] = {"temperature": jnp.array([[1.0]])}
    graph = jraph.GraphsTuple(**graph)

    packer = _packing.Pack(
        attrs={"globals.temperature": base.Attr("0e")},
        out_field="nodes.attributes",
    )

    params = packer.init(rng_key, graph)
    graph_out = packer.apply(params, graph)

    assert "attributes" in graph_out.nodes
    assert isinstance(graph_out.nodes["attributes"], e3j.IrrepsArray)
    assert graph_out.nodes["attributes"].irreps == e3j.Irreps("0e")
    # Check shape: should be (n_nodes, 1)
    assert graph_out.nodes["attributes"].shape == (graph.nodes[keys.FEATURES].shape[0], 1)
