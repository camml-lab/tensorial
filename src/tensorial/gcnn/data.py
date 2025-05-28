import collections
from collections.abc import Iterable, Iterator, Sequence
import enum
import functools
import itertools
from typing import Any, Optional, Union

import beartype
import jax
import jaxtyping as jt
import jraph
import numpy as np
from pytray import tree

from . import keys, utils
from .. import data
from .. import utils as tensorial_utils


class GraphBatch(tuple):
    inputs: jraph.GraphsTuple
    targets: Optional[Any]


GraphDataset = data.Dataset[GraphBatch]
GraphPadding = collections.namedtuple("GraphPadding", ["n_nodes", "n_edges", "n_graphs"])


def max_padding(*padding: GraphPadding) -> "GraphPadding":
    """Get a padding that contains the maximum number of nodes, edges and graphs over all the
    provided paddings"""
    n_node = 0
    n_edge = 0
    n_graph = 0
    for pad in padding:
        n_node = max(n_node, pad.n_nodes)
        n_edge = max(n_edge, pad.n_edges)
        n_graph = max(n_graph, pad.n_graphs)
    return GraphPadding(n_node, n_edge, n_graph)


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
    mask_field=keys.MASK,
    what=GraphAttributes.ALL,
    overwrite=False,
    np_=None,
) -> jraph.GraphsTuple:
    """
    Add a mask array to the ``mask_field`` of ``graph`` for either nodes, edges and/or globals which
    can be used to determine which entries are there just for padding (and therefore should be
    ignored in any computations).

    If ``overwrite`` is ``True`` then any mask already found in the mask field will be overwritten
    by the padding mask. Otherwise, it will be ANDed.
    """
    if np_ is None:
        np_ = tensorial_utils.infer_backend(jax.tree.leaves(graph))

    mask_path = utils.path_from_str(mask_field)
    updates = utils.UpdateDict(graph._asdict())

    # Create the masks that we have been asked to add
    masks = {}
    if what & GraphAttributes.NODES:
        mask = jraph.get_node_padding_mask(graph)
        if not isinstance(mask, np_.ndarray):
            mask = np_.array(mask)
        masks["nodes"] = mask

    if what & GraphAttributes.EDGES:
        mask = jraph.get_edge_padding_mask(graph)
        if not isinstance(mask, np_.ndarray):
            mask = np_.array(mask)
        masks["edges"] = mask

    if what & GraphAttributes.GLOBALS:
        mask = jraph.get_graph_padding_mask(graph)
        if not isinstance(mask, np_.ndarray):
            mask = np_.array(mask)
        masks["globals"] = mask

    for key, mask in masks.items():
        path = (key,) + mask_path
        if not overwrite:
            try:
                mask = mask & tree.get_by_path(updates, path)
            except KeyError:
                pass

        tree.set_by_path(updates, path, mask)

    return jraph.GraphsTuple(**updates._asdict())


def pad_with_graphs(
    graph: jraph.GraphsTuple,
    n_node: int,
    n_edge: int,
    n_graph: int = 2,
    mask_field=keys.MASK,
    overwrite_mask=False,
) -> jraph.GraphsTuple:
    padded = jraph.pad_with_graphs(graph, n_node, n_edge, n_graph)
    padded = add_padding_mask(padded, mask_field=mask_field, overwrite=overwrite_mask)
    return padded


class GraphLoader(data.DataLoader[tuple[jraph.GraphsTuple, ...]]):
    """Data loader for graphs"""

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        *datasets: Optional[Union[jraph.GraphsTuple, Sequence[jraph.GraphsTuple]]],
        batch_size: int = 1,
        shuffle: bool = False,
        pad: Optional[bool] = None,
        padding: Optional[GraphPadding] = None,
    ):
        # Params
        self._batch_size = batch_size
        self._shuffle = shuffle

        # State
        # If the graphs were supplied as GraphTuples then unbatch them to have a base sequence of
        # individual graphs per input
        unbatched = tuple(
            jraph.unbatch_np(graphs) if isinstance(graphs, jraph.GraphsTuple) else graphs
            for graphs in datasets
        )

        # Find one that is not None that we can use to generate the sequence sampler
        example = next(filter(lambda g: g is not None, unbatched))
        self._sampler: data.Sampler[list[int]] = data.samplers.create_sequence_sampler(
            example, batch_size=batch_size, shuffle=shuffle
        )

        if pad is None:
            pad = padding is not None

        create_batcher = functools.partial(
            GraphBatcher, batch_size=batch_size, shuffle=shuffle, pad=pad, padding=padding
        )
        self._batchers: tuple[Optional[GraphBatcher], ...] = tuple(
            create_batcher(graph_batch) if graph_batch is not None else None
            for graph_batch in unbatched
        )

    @property
    def padding(self) -> GraphPadding:
        return self._batchers[0].padding

    def __len__(self) -> int:
        return len(self._sampler)

    def __iter__(self) -> Iterator[tuple[jraph.GraphsTuple, ...]]:
        for idxs in self._sampler:
            batch_graphs = tuple(
                batcher.fetch(idxs) if batcher is not None else None for batcher in self._batchers
            )
            yield batch_graphs


class GraphBatcher(Iterable[jraph.GraphsTuple]):
    """
    Take an iterable of graphs tuples and break it up into batches
    """

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        graphs: Union[jraph.GraphsTuple, Sequence[jraph.GraphsTuple]],
        batch_size: int = 1,
        *,
        shuffle=False,
        pad=False,
        add_mask=True,
        padding: Optional[GraphPadding] = None,
    ):
        if isinstance(graphs, jraph.GraphsTuple):
            graphs = jraph.unbatch_np(graphs)
        else:
            for graph in graphs:
                if len(graph.n_node) != 1:
                    raise ValueError("``graphs`` should be a sequence of individual graphs")

        self._graphs = graphs
        self._batch_size = batch_size
        self._add_mask = add_mask
        self._sampler = data.samplers.create_batch_sampler(
            self._graphs, batch_size=batch_size, shuffle=shuffle
        )

        self._padding = (
            None
            if not pad
            else (
                padding
                if padding is not None
                else self.calculate_padding(graphs, batch_size, with_shuffle=shuffle)
            )
        )

    @staticmethod
    def calculate_padding(
        graphs: Sequence[jraph.GraphsTuple], batch_size: int, with_shuffle: bool = False
    ) -> GraphPadding:
        """Calculate the padding necessary to fit the given graphs into a batch"""
        if with_shuffle:
            # Calculate the maximum possible number of nodes and edges over any possible shuffling
            pad_nodes = (
                sum(sorted([graph.n_node[0] for graph in graphs], reverse=True)[:batch_size]) + 1
            )
            pad_edges = sum(
                sorted([graph.n_edge[0] for graph in graphs], reverse=True)[:batch_size]
            )
        else:
            pad_nodes = 0
            pad_edges = 0

            for batch in _chunks(graphs, batch_size):
                pad_nodes = max(pad_nodes, sum(graph.n_node.item() for graph in batch))
                pad_edges = max(pad_edges, sum(graph.n_edge.item() for graph in batch))
            pad_nodes += 1

        return GraphPadding(pad_nodes, pad_edges, n_graphs=batch_size + 1)

    @property
    def padding(self) -> GraphPadding:
        return self._padding

    def fetch(self, idxs: Sequence[int]) -> jraph.GraphsTuple:
        if len(idxs) > self._batch_size:
            raise ValueError(
                f"Number of indices must be less than or equal to the batch size "
                f"({self._batch_size}), got {len(idxs)}"
            )
        batch = _fetch_batch(self._graphs, idxs)
        if self._padding is not None:
            batch = jraph.pad_with_graphs(batch, *self._padding)
            if self._add_mask:
                batch = add_padding_mask(batch)

        return batch

    def __iter__(self) -> Iterator[jraph.GraphsTuple]:
        for idxs in self._sampler:
            yield self.fetch(idxs)


def _fetch_batch(
    graphs: Sequence[jraph.GraphsTuple], idxs: Sequence[int], np_=np
) -> jraph.GraphsTuple:
    """Given a set of indices, fetch the corresponding batch from the given graphs."""
    batch = []
    for idx in idxs:
        batch.append(graphs[idx])

    if np_ is np:
        return jraph.batch_np(batch)

    # Assume JNP
    return jraph.batch(batch)


def get_by_path(graph: jraph.GraphsTuple, path: tuple, pad_value=None) -> Any:
    res = tree.get_by_path(graph._asdict(), path)
    np_ = tensorial_utils.infer_backend(res)

    if pad_value is not None:
        mask = np_.ones(res.shape[0], dtype=bool)
        if path[0] == "globals":
            mask = jraph.get_graph_padding_mask(graph)
        elif path[0] == "edges":
            mask = jraph.get_edge_padding_mask(graph)
        elif path[0] == "nodes":
            mask = jraph.get_node_padding_mask(graph)
        res = np_.where(mask, res, pad_value)

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


def _chunks(iterable: Iterable, batch_size: int):
    "Collect data into non-overlapping fixed-length chunks or blocks."
    it = iter(iterable)
    while chunk := list(itertools.islice(it, batch_size)):
        yield chunk
