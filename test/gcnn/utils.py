import random
from typing import Sequence

import e3nn_jax as e3j
import jraph
import numpy as np

from tensorial import gcnn
import tensorial.modules


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
) -> tensorial.modules.Sequential:
    return tensorial.modules.Sequential(
        [
            gcnn.NodewiseEncoding(attrs={gcnn.keys.SPECIES: tensorial.OneHot(len(type_numbers))}),
            gcnn.EdgeVectors(),
            gcnn.EdgewiseEncoding(
                attrs=dict(
                    edge_vectors=tensorial.SphericalHarmonic(irreps="0e + 1o + 2e", normalise=True)
                )
            ),
            gcnn.RadialBasisEdgeEncoding(r_max=r_max),
            gcnn.NodewiseLinear(node_feature_irreps, field=gcnn.keys.ATTRIBUTES),
            *modules,
        ]
    )
