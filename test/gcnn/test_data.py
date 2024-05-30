# -*- coding: utf-8 -*-
import jraph
import numpy as np

from tensorial.gcnn import data, keys


def test_add_padding_mask(cube_graph: jraph.GraphsTuple):
    mask = np.zeros(dtype=bool, shape=(len(cube_graph.nodes[keys.POSITIONS]),))
    mask[0] = True
    cube_graph.nodes[keys.MASK] = mask
    padded = jraph.pad_with_graphs(
        cube_graph, n_node=cube_graph.n_node.item() + 1, n_edge=cube_graph.n_edge.item() + 1
    )
    padded = data.add_padding_mask(padded)

    assert np.all(padded.nodes[keys.MASK][:-1] == mask)
    assert padded.nodes[keys.MASK][-1].item() is False

    # Test overwrite
    padded = data.add_padding_mask(padded, overwrite=True)
    assert np.all(padded.nodes[keys.MASK] == np.array([*[True] * cube_graph.n_node.item(), False]))
