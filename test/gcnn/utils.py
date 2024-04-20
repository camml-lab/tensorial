# -*- coding: utf-8 -*-
import random

import jraph
import numpy as np

from tensorial import gcnn


def random_spatial_graph(num_nodes: int = None, cutoff=0.2) -> jraph.GraphsTuple:
    if num_nodes is None:
        num_nodes = random.randint(2, 10)
    pos = np.random.rand(num_nodes, 3)
    return gcnn.graph_from_points(pos, r_max=cutoff)
