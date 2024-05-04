# -*- coding: utf-8 -*-
import enum
from typing import Any, Iterable, Iterator, Optional, Tuple, Union

import jax.numpy as jnp
import jraph
from pytray import tree

from tensorial import datasets

from . import keys, utils


class GraphBatch(datasets.Batch):
    inputs: jraph.GraphsTuple
    targets: Optional[Any]


GraphDataset = Iterable[GraphBatch]


class GraphAttributes(enum.IntFlag):
    NODES = 0b0001
    EDGES = 0b0010
    GLOBALS = 0b0100
    ALL = NODES | EDGES | GLOBALS


def generated_padded_graphs(
    dataset: GraphDataset,
    add_mask=False,
    num_nodes=None,
    num_edges=None,
    num_graphs=None,
) -> Iterator[GraphBatch]:
    """
    Provides an iterator over graphs tuple batches that are padded to make the number of nodes,
    edges and graphs in each batch equal to the maximum found in the dataset
    """
    if None in (num_nodes, num_edges, num_graphs):
        # We have to calculate a maximum for one or more of the padding numbers
        max_nodes = 0
        max_edges = 0
        max_graphs = 0
        for batch_in, _output in dataset:
            max_nodes = max(max_nodes, sum(batch_in.n_node))
            max_edges = max(max_edges, sum(batch_in.n_edge))
            max_graphs = max(max_graphs, len(batch_in.n_node))

        num_nodes = max_nodes + 1 if num_nodes is None else num_nodes
        num_edges = max_edges if num_edges is None else num_edges
        num_graphs = max_graphs + 1 if num_graphs is None else num_graphs

    for batch_in, batch_out in dataset:
        if isinstance(batch_in, jraph.GraphsTuple):
            batch_in = jraph.pad_with_graphs(batch_in, num_nodes, num_edges, num_graphs)
            if add_mask:
                batch_in = add_padding_mask(batch_in)

        if isinstance(batch_out, jraph.GraphsTuple):
            batch_out = jraph.pad_with_graphs(batch_out, num_nodes, num_edges, num_graphs)
            if add_mask:
                batch_out = add_padding_mask(batch_out)

        yield GraphBatch(batch_in, batch_out)


def add_padding_mask(
    graph: jraph.GraphsTuple,
    mask_field=keys.DEFAULT_PAD_MASK_FIELD,
    what=GraphAttributes.ALL,
    overwrite=False,
) -> jraph.GraphsTuple:
    """
    Add a mask array to the `mask_field` of `graph` for either nodes, edges and/or globals which
    can be used to determine which entries are there just for padding (and therefore should be
    ignored in any computations).

    If `overwrite` is `True` then any mask already found in the mask field will be overwritten by
    the padding mask. Otherwise, it will be ORed.
    """
    mask_path = utils.path_from_str(mask_field)
    updates = utils.UpdateDict(graph._asdict())

    # Create the masks that we have been asked to add
    masks = {}
    if what & GraphAttributes.NODES:
        masks["nodes"] = jraph.get_node_padding_mask(graph)
    if what & GraphAttributes.EDGES:
        masks["edges"] = jraph.get_edge_padding_mask(graph)
    if what & GraphAttributes.GLOBALS:
        masks["globals"] = jraph.get_graph_padding_mask(graph)

    for key, mask in masks.items():
        path = (key,) + mask_path
        if not overwrite:
            try:
                mask = mask | tree.get_by_path(updates, path)
            except KeyError:
                pass

        tree.set_by_path(updates, path, mask)

    return jraph.GraphsTuple(**updates._asdict())


def generate_batches(
    batch_size: int,
    inputs: Union[jraph.GraphsTuple, Iterable[jraph.GraphsTuple]],
    outputs: Union[jraph.GraphsTuple, Iterable] = None,
    batch_builder=jraph.batch,
    output_batch_builder=tuple,
) -> Iterator[datasets.Batch]:
    """Specialised version of batch creator for graph data"""
    if isinstance(inputs, jraph.GraphsTuple):
        # The user has supplied a graphs tuple as the input so assume this is already batched
        inputs = jraph.unbatch(inputs)
    if isinstance(outputs, jraph.GraphsTuple):
        outputs = jraph.unbatch(outputs)
        output_batch_builder = jraph.batch

    return datasets.generate_batches(
        batch_size, inputs, outputs, batch_builder, output_batch_builder
    )


def get_by_path(graph: jraph.GraphsTuple, path: Tuple, pad_value=None) -> Any:
    res = tree.get_by_path(graph._asdict(), path)
    if pad_value is not None:
        mask = jnp.ones(res.shape[0], dtype=bool)
        if path[0] == "globals":
            mask = jraph.get_graph_padding_mask(graph)
        elif path[0] == "edges":
            mask = jraph.get_edge_padding_mask(graph)
        elif path[0] == "nodes":
            mask = jraph.get_node_padding_mask(graph)
        res = jnp.where(mask, res, pad_value)
    return res
