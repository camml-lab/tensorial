# -*- coding: utf-8 -*-
import enum
import functools
from typing import Any, Iterator, Optional, Sequence, Tuple, Union

import jax.numpy as jnp
import jraph
import numpy as np
from pytray import tree

import tensorial
from tensorial import data

from . import keys, utils


class GraphBatch(tuple):
    inputs: jraph.GraphsTuple
    targets: Optional[Any]


GraphDataset = tensorial.data.Dataset[GraphBatch]


class GraphAttributes(enum.IntFlag):
    NODES = 0b0001
    EDGES = 0b0010
    GLOBALS = 0b0100
    ALL = NODES | EDGES | GLOBALS


def generated_padded_graphs(
    dataset: GraphDataset, add_mask=False, num_nodes=None, num_edges=None, num_graphs=None
) -> Iterator[GraphBatch]:
    """
    Provides an iterator over graphs tuple batches that are padded to make the number of nodes,
    edges and graphs in each batch equal to the maximum found in the dataset
    """
    if None in (num_nodes, num_edges, num_graphs):
        # We have o calculate a maximum for one or more of the padding numbers
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
    Add a mask array to the ``mask_field`` of ``graph`` for either nodes, edges and/or globals which
    can be used to determine which entries are there just for padding (and therefore should be
    ignored in any computations).

    If ``overwrite`` is ``True`` then any mask already found in the mask field will be overwritten
    by the padding mask. Otherwise, it will be ORed.
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


class GraphLoader(data.DataLoader[Tuple[jraph.GraphsTuple, ...]]):

    def __init__(
        self,
        *graphs: Optional[Sequence[jraph.GraphsTuple]],
        batch_size: int = 1,
        shuffle: bool = False,
        pad=False,
    ):
        # If the graphs were supplied as GraphTuples then unbatch them to have a base sequence of
        # individual graphs per input
        self._graphs = tuple(
            jraph.unbatch(graph) if isinstance(graph, jraph.GraphsTuple) else graph
            for graph in graphs
        )
        self._sampler = data.samplers.create_sequence_sampler(
            self._graphs[0], batch_size=batch_size, shuffle=shuffle
        )
        self._batch_size = batch_size
        self._shuffle = shuffle

        create_batcher = functools.partial(
            GraphBatcher, batch_size=batch_size, shuffle=shuffle, pad=pad
        )
        self._batchers = tuple(
            create_batcher(graph_batch) if graph_batch is not None else None
            for graph_batch in graphs
        )

    def __len__(self) -> int:
        return len(self._sampler)

    def __iter__(self) -> Iterator[Tuple[jraph.GraphsTuple, ...]]:
        for idxs in self._sampler:
            batch_graphs = tuple(
                batcher.fetch(idxs) if batcher is not None else None for batcher in self._batchers
            )
            yield batch_graphs


class GraphBatcher:

    def __init__(
        self,
        graphs: Union[jraph.GraphsTuple, Sequence[jraph.GraphsTuple]],
        batch_size: int = 1,
        shuffle=False,
        pad=False,
        add_mask=True,
    ):
        if isinstance(graphs, jraph.GraphsTuple):
            graphs = jraph.unbatch(graphs)
        else:
            for graph in graphs:
                if len(graph.n_node) != 1:
                    raise ValueError("``graphs`` should be a sequence of individual graphs")

        self._graphs = graphs
        self._batch_size = batch_size
        self._pad = pad
        self._add_mask = add_mask
        self._sampler = data.samplers.create_batch_sampler(
            self._graphs, batch_size=batch_size, shuffle=shuffle
        )

        if shuffle:
            # Calculate the maximum possible number of nodes and edges over any possible shuffling
            self._pad_nodes = (
                sum(sorted([graph.n_node[0] for graph in graphs], reverse=True)[:batch_size]) + 1
            )
            self._pad_edges = sum(
                sorted([graph.n_edge[0] for graph in graphs], reverse=True)[:batch_size]
            )
        else:
            # Here we just want the most nodes and edges in any of the batches
            self._pad_nodes = max(self._fetch(idxs).n_node.sum() for idxs in self._sampler) + 1
            self._pad_edges = max(self._fetch(idxs).n_edge.sum() for idxs in self._sampler)

        self._pad_graphs = batch_size + 1

    def fetch(self, idxs: Sequence[int]) -> jraph.GraphsTuple:
        batch = self._fetch(idxs)
        if self._pad:
            batch = jraph.pad_with_graphs(
                batch, n_node=self._pad_nodes, n_edge=self._pad_edges, n_graph=self._pad_graphs
            )
            if self._add_mask:
                batch = add_padding_mask(batch)

        return batch

    def _fetch(self, idxs: Sequence[int]) -> jraph.GraphsTuple:
        if len(idxs) > self._batch_size:
            raise ValueError(
                f"Number of indices must be less than or equal to the batch size "
                f"({self._batch_size}), got {len(idxs)}"
            )

        batch = []
        for idx in idxs:
            batch.append(self._graphs[idx])
        return jraph.batch(batch)

    def __iter__(self) -> Iterator[jraph.GraphsTuple]:
        for idxs in self._sampler:
            yield self.fetch(idxs)


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


def get_graph_stats(*graph: jraph.GraphsTuple) -> dict:
    nodes = np.array([len(g.n_node) for g in graph])
    edges = np.array([len(g.n_edge) for g in graph])

    return dict(
        min_nodes=nodes.min,
        max_nodes=nodes.max(),
        mean_nodes=nodes.mean(),
        min_edges=edges.min(),
        max_edges=edges.max(),
        avg_edges=edges.mean(),
    )
