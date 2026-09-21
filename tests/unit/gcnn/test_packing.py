import e3nn_jax as e3j
import jax.numpy as jnp
import jraph
import pytest

from tensorial import base
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


def test_pack_multiple_node_attrs(cube_graph_gcnn: jraph.GraphsTuple, rng_key):
    # Concatenate two node attributes (species + positions) into a single packing
    graph = cube_graph_gcnn
    n_nodes = graph.nodes[keys.SPECIES].shape[0]

    packer = _packing.Pack(
        attrs={
            "nodes.species": base.Attr("1x0e"),
            "nodes.positions": base.Attr("1o"),
        },
        out_field="nodes.attributes",
    )
    params = packer.init(rng_key, graph)
    graph_out = packer.apply(params, graph)

    assert isinstance(graph_out.nodes["attributes"], e3j.IrrepsArray)
    assert graph_out.nodes["attributes"].irreps == e3j.Irreps("1x0e + 1o")
    assert graph_out.nodes["attributes"].shape == (n_nodes, 4)


def test_pack_whole_component(cube_graph_gcnn: jraph.GraphsTuple, rng_key):
    # ``attrs`` may describe a whole component at once via an IrrepsObj type
    class NodeEmbed(base.IrrepsObj):
        species = base.Attr("1x0e")
        positions = base.Attr("1o")

    graph = cube_graph_gcnn
    n_nodes = graph.nodes[keys.SPECIES].shape[0]

    packer = _packing.Pack(attrs=NodeEmbed, out_field="nodes.embedded")
    params = packer.init(rng_key, graph)
    graph_out = packer.apply(params, graph)

    assert isinstance(graph_out.nodes["embedded"], e3j.IrrepsArray)
    assert graph_out.nodes["embedded"].irreps == e3j.Irreps("1x0e + 1o")
    assert graph_out.nodes["embedded"].shape == (n_nodes, 4)


def test_pack_globals_into_edges(cube_graph_gcnn: jraph.GraphsTuple, rng_key):
    # A per-graph global is broadcast across the edges of the destination component
    graph_dict = cube_graph_gcnn._asdict()
    graph_dict["globals"] = {"temperature": jnp.array([[5.0]])}
    graph = jraph.GraphsTuple(**graph_dict)
    n_edges = len(cube_graph_gcnn.senders)

    packer = _packing.Pack(
        attrs={"globals.temperature": base.Attr("0e")},
        out_field="edges.attributes",
    )
    params = packer.init(rng_key, graph)
    graph_out = packer.apply(params, graph)

    assert isinstance(graph_out.edges["attributes"], e3j.IrrepsArray)
    assert graph_out.edges["attributes"].irreps == e3j.Irreps("0e")
    assert graph_out.edges["attributes"].shape == (n_edges, 1)
    assert jnp.all(base.as_array(graph_out.edges["attributes"]) == 5.0)


def test_pack_globals_into_globals(cube_graph_gcnn: jraph.GraphsTuple, rng_key):
    # A global packed into the ``globals`` component is not broadcast at all
    graph_dict = cube_graph_gcnn._asdict()
    graph_dict["globals"] = {"temperature": jnp.array([[5.0]])}
    graph = jraph.GraphsTuple(**graph_dict)

    packer = _packing.Pack(
        attrs={"globals.temperature": base.Attr("0e")},
        out_field="globals.attributes",
    )
    params = packer.init(rng_key, graph)
    graph_out = packer.apply(params, graph)

    assert isinstance(graph_out.globals["attributes"], e3j.IrrepsArray)
    assert graph_out.globals["attributes"].irreps == e3j.Irreps("0e")
    assert graph_out.globals["attributes"].shape == (1, 1)
    assert jnp.all(base.as_array(graph_out.globals["attributes"]) == 5.0)


def test_pack_scalar_global_broadcasts_to_nodes(cube_graph_gcnn: jraph.GraphsTuple, rng_key):
    # A 1-D global (ndim < 2) is broadcast across the destination's leading axis
    graph_dict = cube_graph_gcnn._asdict()
    graph_dict["globals"] = {"temperature": jnp.array([5.0])}
    graph = jraph.GraphsTuple(**graph_dict)
    n_nodes = graph.nodes[keys.SPECIES].shape[0]

    packer = _packing.Pack(
        attrs={"globals.temperature": base.Attr("0e")},
        out_field="nodes.attributes",
    )
    params = packer.init(rng_key, graph)
    graph_out = packer.apply(params, graph)

    assert graph_out.nodes["attributes"].shape == (n_nodes, 1)
    assert jnp.all(base.as_array(graph_out.nodes["attributes"]) == 5.0)


def test_pack_broadcast_uses_sum_when_shape_from_none(cube_graph_gcnn: jraph.GraphsTuple, rng_key):
    # With ``shape_from=None`` the broadcast length is derived from ``jnp.sum(n_node)``
    graph_dict = cube_graph_gcnn._asdict()
    graph_dict["globals"] = {"temperature": jnp.array([[5.0]])}
    graph = jraph.GraphsTuple(**graph_dict)
    n_nodes = int(jnp.sum(cube_graph_gcnn.n_node))

    packer = _packing.Pack(
        attrs={"globals.temperature": base.Attr("0e")},
        out_field="nodes.attributes",
        shape_from=None,
    )
    params = packer.init(rng_key, graph)
    graph_out = packer.apply(params, graph)

    assert graph_out.nodes["attributes"].shape == (n_nodes, 1)
    assert jnp.all(base.as_array(graph_out.nodes["attributes"]) == 5.0)


def test_pack_rejects_invalid_out_field(cube_graph_gcnn: jraph.GraphsTuple, rng_key):
    packer = _packing.Pack(
        attrs={"nodes.species": base.Attr("1x0e")},
        out_field="cells.attributes",
    )
    with pytest.raises(ValueError, match="must start with one of"):
        packer.init(rng_key, cube_graph_gcnn)


def test_pack_rejects_invalid_attr_path(cube_graph_gcnn: jraph.GraphsTuple, rng_key):
    packer = _packing.Pack(
        attrs={"bad.path": base.Attr("1x0e")},
        out_field="nodes.attributes",
    )
    with pytest.raises(ValueError, match="must start with one of"):
        packer.init(rng_key, cube_graph_gcnn)


def test_pack_rejects_missing_source_attr(cube_graph_gcnn: jraph.GraphsTuple, rng_key):
    packer = _packing.Pack(
        attrs={"nodes.does_not_exist": base.Attr("1x0e")},
        out_field="nodes.attributes",
    )
    with pytest.raises(ValueError, match="Did not find"):
        packer.init(rng_key, cube_graph_gcnn)
