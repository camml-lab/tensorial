from typing import Optional

import clu.metrics
from clu.metrics import Metric
from jax import random
import jax.numpy as jnp
import jraph
import optax

from tensorial import gcnn
from tensorial.gcnn import metrics


class Std(clu.metrics.Std):
    @classmethod
    def from_model_output(  # pylint: disable=arguments-differ
        cls,
        *,
        values: jnp.array,
        mask: Optional[jnp.array] = None,
    ) -> Metric:
        return clu.metrics.Std.from_model_output(jnp.astype(values, jnp.float32), mask=mask)


def test_graph_metric(cube_graph: jraph.GraphsTuple, rng_key):
    rng_key, *keys = random.split(rng_key, 4)

    # Let's add some data to the nodes
    nodes = cube_graph.nodes
    nodes["sizes"] = random.uniform(keys[0], (cube_graph.n_node[0],))

    node_sizes_avg = metrics.graph_metric(clu.metrics.Average, "graph.nodes.sizes")
    res = node_sizes_avg.from_model_output(graph=cube_graph)
    assert jnp.isclose(res.compute(), nodes["sizes"].mean())

    edges = cube_graph.edges
    edges["energies"] = random.uniform(keys[1], (cube_graph.n_edge[0],))
    # This time use kwargs to graph_metric which will be passed to Std by kwargs this time
    # (instead of args)
    edge_length_std = metrics.graph_metric(Std, values="graph.edges.energies")
    res = edge_length_std.from_model_output(graph=cube_graph)
    assert jnp.isclose(res.compute(), edges["energies"].std())

    # Let's try a metric that takes two arguments,
    # see: https://optax.readthedocs.io/en/latest/api/losses.html#optax.squared_error
    nodes["size_predictions"] = random.uniform(keys[0], (cube_graph.n_node[0],))
    size_mse = metrics.graph_metric(
        clu.metrics.Average.from_fun(optax.squared_error),
        predictions="graph.nodes.size_predictions",
        targets="graph.nodes.sizes",
    )
    res = size_mse.from_model_output(graph=cube_graph)
    assert jnp.isclose(
        res.compute(),
        optax.squared_error(nodes["size_predictions"], nodes["sizes"]).mean(),
    )


def test_graph_metric_per_node(rng_key):
    """Test the per node normalisation of graph metrics"""
    # Create some random graphs
    random_graphs = jraph.batch(tuple(gcnn.random.spatial_graph(rng_key) for _ in range(10)))
    random_graphs.globals["num_nodes"] = random_graphs.n_node

    # Without normalisation by num nodes
    node_sizes_avg = metrics.graph_metric(clu.metrics.Average, "graph.globals.num_nodes")
    metr = node_sizes_avg.from_model_output(graph=random_graphs)
    assert jnp.isclose(metr.compute(), jnp.mean(random_graphs.globals["num_nodes"]))

    # ...and with
    node_sizes_avg = metrics.graph_metric(
        clu.metrics.Average, "graph.globals.num_nodes", _per_node=True
    )
    metr = node_sizes_avg.from_model_output(graph=random_graphs)
    assert jnp.isclose(metr.compute(), 1.0)

    node_sizes_avg = metrics.graph_metric(
        clu.metrics.Average,
        "graph.globals.num_nodes",
        mask=("globals", gcnn.keys.MASK),
        _per_node=True,
    )
    padded = tuple(gcnn.data.GraphBatcher(random_graphs, pad=True, add_mask=True))[0]
    metr = node_sizes_avg.from_model_output(graph=padded)
    assert jnp.isclose(metr.compute(), 1.0)
