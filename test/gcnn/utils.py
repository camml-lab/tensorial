from collections.abc import Sequence
import random

import e3nn_jax as e3j
import jraph
import numpy as np

from tensorial import gcnn
import tensorial.nn


def random_spatial_graph(num_nodes: int = None, cutoff=0.2) -> jraph.GraphsTuple:
    if num_nodes is None:
        num_nodes = random.randint(2, 10)
    pos = np.random.rand(num_nodes, 3)
    return gcnn.graph_from_points(pos, r_max=cutoff)


def graph_model(
    r_max: float,
    node_feature_irreps: e3j.Irreps,
    *modules,
    type_numbers: Sequence[int] = (0,),
) -> tensorial.nn.Sequential:
    return tensorial.nn.Sequential(
        [
            gcnn.NodewiseEmbedding(attrs={gcnn.keys.SPECIES: tensorial.OneHot(len(type_numbers))}),
            gcnn.EdgeVectors(),
            gcnn.EdgewiseEmbedding(
                attrs=dict(
                    edge_vectors=tensorial.SphericalHarmonic(irreps="0e + 1o + 2e", normalise=True)
                )
            ),
            gcnn.RadialBasisEdgeEmbedding(r_max=r_max),
            gcnn.NodewiseLinear(node_feature_irreps, field=gcnn.keys.ATTRIBUTES),
            *modules,
        ]
    )
