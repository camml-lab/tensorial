from collections.abc import Iterable, Iterator, Sequence
import itertools
import logging
from typing import TYPE_CHECKING

import beartype
import jax
import jaxtyping as jt
import jraph
import numpy as np
from pytray import tree
import reax

from . import _common, _graph_padding
from .. import keys, utils
from ... import utils as tensorial_utils

if TYPE_CHECKING:
    import tensorial.gcnn.data


_LOGGER = logging.getLogger(__name__)


__all__ = (
    "max_padding",
    "generated_padded_graphs",
    "add_padding_mask",
    "pad_with_graphs",
    "GraphBatcher",
)


def max_padding(*padding: "tensorial.gcnn.data.GraphPadding") -> "tensorial.gcnn.data.GraphPadding":
    """Get a padding that contains the maximum number of nodes, edges and graphs over all the
    provided paddings
    """
    n_node = 0
    n_edge = 0
    n_graph = 0
    for pad in padding:
        n_node = max(n_node, pad.n_nodes)
        n_edge = max(n_edge, pad.n_edges)
        n_graph = max(n_graph, pad.n_graphs)
    return _common.GraphPadding(n_node, n_edge, n_graph)


def generated_padded_graphs(
    dataset: "tensorial.gcnn.data.GraphDataset",
    add_mask=False,
    num_nodes=None,
    num_edges=None,
    num_graphs=None,
) -> "Iterator[tensorial.gcnn.data.GraphBatch]":
    """Provides an iterator over graphs tuple batches that are padded to make the number of nodes,
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

        yield _common.GraphBatch((batch_in, batch_out))


def add_padding_mask(
    graph: jraph.GraphsTuple,
    mask_field=keys.MASK,
    what=_common.GraphAttributes.ALL,
    overwrite=False,
    np_=None,
) -> jraph.GraphsTuple:
    """Add a mask array to the ``mask_field`` of ``graph`` for either nodes, edges and/or globals
    which can be used to determine which entries are there just for padding (and therefore should be
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
    if what & _common.GraphAttributes.NODES:
        mask = _graph_padding.get_node_padding_mask(graph, np_=np_)
        masks["nodes"] = mask

    if what & _common.GraphAttributes.EDGES:
        mask = _graph_padding.get_edge_padding_mask(graph, np_=np_)
        masks["edges"] = mask

    if what & _common.GraphAttributes.GLOBALS:
        mask = _graph_padding.get_graph_padding_mask(graph, np_=np_)
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
    mask_field: str | None = keys.MASK,
    overwrite_mask=False,
) -> jraph.GraphsTuple:
    padded = jraph.pad_with_graphs(graph, n_node, n_edge, n_graph)
    if mask_field:
        padded = add_padding_mask(padded, mask_field=mask_field, overwrite=overwrite_mask)
    return padded


def _get_pad_multiple(pad_to_multiple: int | str | jax.Device | None) -> int | None:
    if pad_to_multiple is None:
        return None

    if isinstance(pad_to_multiple, int):
        return pad_to_multiple

    platform = None
    if isinstance(pad_to_multiple, jax.Device):
        platform = pad_to_multiple.platform
    elif pad_to_multiple == "auto":
        platform = jax.devices()[0].platform
    elif isinstance(pad_to_multiple, str):
        platform = pad_to_multiple.lower()

    if platform == "tpu":
        return 128
    if platform in ("gpu", "cuda", "rocm"):
        return 64
    if platform == "cpu":
        return None

    raise ValueError(f"Unknown padding strategy or platform for pad_to_multiple: {pad_to_multiple}")


class GraphBatcher(Iterable[jraph.GraphsTuple]):
    """Take an iterable of graph tuples and break it up into batches.

    Iterates over batches of :class:`jraph.GraphsTuple`, optionally padding each batch to a
    fixed size and adding padding masks so that downstream code can tell which entries are
    real and which are there just for padding.

    Two batching modes are supported:

    * ``implicit`` (default) — graphs are concatenated into a single flat batch with a
      single padding graph at the end, the standard jraph batching scheme.
    * ``explicit`` — each graph is padded to the maximum size and stacked, so that
      ``batch.n_node`` has shape ``(batch_size, 2)``. This keeps a static shape per graph
      and is useful for certain model architectures.

    Args:
        graphs: a sequence of individual (unbatched) graphs, or a single batched graph
        batch_size: the number of graphs per batch
        shuffle: if ``True``, shuffle the graphs before batching
        pad: if ``True``, pad each batch to the calculated (or supplied) padding size
        add_mask: if ``True`` and ``pad``, add padding masks to the nodes, edges and
            globals of each batch
        padding: an explicit :class:`~tensorial.gcnn.data.GraphPadding` to use. If ``None``
            and ``pad`` is ``True`` in implicit mode, the padding is calculated
            automatically
        pad_to_multiple: round the padding up to a multiple of this value. Can be an
            integer, a device (e.g. ``jax.devices()[0]``), or the string ``"auto"`` to
            choose a platform-appropriate value
        drop_last: if ``True``, drop the last incomplete batch
        mode: the batching mode, either ``"implicit"`` or ``"explicit"``

    Example:
        >>> batcher = GraphBatcher(graphs, batch_size=8, pad=True)
        >>> for batch in batcher:
        ...     process(batch)
    """

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        graphs: jraph.GraphsTuple | Sequence[jraph.GraphsTuple],
        batch_size: int = 1,
        *,
        shuffle: bool = False,
        pad: bool = False,
        add_mask: bool = True,
        padding: "tensorial.gcnn.data.GraphPadding | None" = None,
        pad_to_multiple: "int | str | jax.Device | None" = None,
        drop_last: bool = False,
        mode: "str | tensorial.gcnn.data.BatchMode" = _common.BatchMode.IMPLICIT,
    ):
        if add_mask and not pad:
            _LOGGER.warning(
                "User asked for mask to be added but there is no padding "
                "(so we don't know what to mask off).  Ignoring"
            )
            add_mask = False

        # Params
        self._batch_size: int = batch_size
        self._add_mask: bool = add_mask
        self._mode: "tensorial.gcnn.data.BatchMode" = _common.BatchMode(mode)

        if isinstance(graphs, jraph.GraphsTuple):
            graphs = jraph.unbatch_np(graphs)
        else:
            for graph in graphs:
                if len(graph.n_node) != 1:
                    raise ValueError("``graphs`` should be a sequence of individual graphs")

        if self._mode is _common.BatchMode.IMPLICIT:
            if pad and padding is None:
                # Automatically determine padding
                padding = self.calculate_padding(
                    graphs, batch_size, with_shuffle=shuffle, pad_to_multiple=pad_to_multiple
                )
        else:  # explicit batching
            # The padding now applies to each graph and not the batches themselves
            batcher = GraphBatcher(
                graphs,
                batch_size=1,
                pad=True,
                add_mask=True,
                padding=padding,
                pad_to_multiple=pad_to_multiple,
                mode=_common.BatchMode.IMPLICIT,
            )
            graphs = list(batcher)
            if padding is None:
                padding = batcher.padding

        self._padding = padding if pad else None
        self._graphs: list[jraph.GraphsTuple] = graphs

        # State
        self._sampler = reax.data.samplers.create_sampler(
            self._graphs, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
        )

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def padding(self) -> "tensorial.gcnn.data.GraphPadding":
        return self._padding

    def __len__(self) -> int:
        return len(self._sampler)

    def __iter__(self) -> Iterator[jraph.GraphsTuple]:
        for idxs in self._sampler:
            yield self.fetch(idxs)

    def __getitem__(self, item):
        return self.fetch(self._sampler[item])

    @staticmethod
    def calculate_padding(
        graphs: Sequence[jraph.GraphsTuple],
        batch_size: int,
        with_shuffle: bool = False,
        pad_to_multiple: int | str | jax.Device | None = None,
    ) -> "tensorial.gcnn.data.GraphPadding":
        """Calculate the padding necessary to fit the given graphs into a batch"""
        # A batch contains at most ``batch_size`` graphs.  Under DDP those
        # graphs are not guaranteed to be ``batch_size`` *distinct* graphs of
        # the dataset (the per-rank block is a misaligned contiguous slice,
        # and its trailing batch can replay a repeated index; see
        # ``reax.data.samplers.DistributedSampler.__iter__``).  The only
        # bound that is safe in every case -- for both the ``shuffle`` and the
        # ``shuffle = False`` (DDP) paths -- is therefore: the sum of the
        # ``batch_size`` largest graphs.  This bounds any batch regardless of
        # whether its elements are distinct or duplicated.
        pad_nodes = sum(sorted([g.n_node.item() for g in graphs], reverse=True)[:batch_size]) + 1
        pad_edges = sum(sorted([g.n_edge.item() for g in graphs], reverse=True)[:batch_size])

        pad_multiple = _get_pad_multiple(pad_to_multiple)
        if pad_multiple is not None:
            pad_nodes = int(np.ceil(pad_nodes / pad_multiple) * pad_multiple)
            pad_edges = int(np.ceil(pad_edges / pad_multiple) * pad_multiple)

        return _common.GraphPadding(pad_nodes, pad_edges, n_graphs=batch_size + 1)

    def fetch(self, idxs: Sequence[int]) -> jraph.GraphsTuple:
        if len(idxs) > self._batch_size:
            raise ValueError(
                f"Number of indices must be less than or equal to the batch size "
                f"({self._batch_size}), got {len(idxs)}"
            )

        if self._mode is _common.BatchMode.IMPLICIT:
            return self._fetch_batch(self._graphs, idxs)

        return self._fetch_batch_explicit(self._graphs, idxs)

    def _fetch_batch(
        self, graphs: Sequence[jraph.GraphsTuple], idxs: Sequence[int], np_=np
    ) -> jraph.GraphsTuple:
        """Given a set of indices, fetch the corresponding batch from the given graphs."""
        graph_list: list[jraph.GraphsTuple] = [graphs[idx] for idx in idxs]

        if np_ is np:
            batch = jraph.batch_np(graph_list)
        else:
            # Assume JAX
            batch = jraph.batch(graph_list)

        if self._padding is not None:
            batch = jraph.pad_with_graphs(batch, *self._padding)
            if self._add_mask:
                batch = add_padding_mask(batch)

        return batch

    def _fetch_batch_explicit(
        self, graphs: Sequence[jraph.GraphsTuple], idxs: Sequence[int], np_=np
    ) -> jraph.GraphsTuple:
        """Given a set of indices, fetch the corresponding batch from the given graphs."""
        graph_list: list[jraph.GraphsTuple] = [graphs[idx] for idx in idxs]
        if len(graph_list) < self._batch_size:
            # We need to add some dummy graphs
            dummy = _dummy_graph_like(graph_list[0])
            graph_list.extend([dummy] * (self._batch_size - len(graph_list)))

        # Perform the stacking of all arrays
        batch = stack_graphs_tuple(graph_list, np_=np_)

        return batch


def _chunks(iterable: Iterable, batch_size: int):
    "Collect data into non-overlapping fixed-length chunks or blocks."
    it = iter(iterable)
    while chunk := list(itertools.islice(it, batch_size)):
        yield chunk


def _dummy_graph_like(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    num_graphs = graph.n_node.shape[0]
    num_nodes = sum(graph.n_node)
    num_edges = sum(graph.n_edge)

    graph_globals = jax.tree.map(np.zeros_like, graph.globals)
    # Set the global mask to False for these graphs as they are dummies
    if isinstance(graph_globals, dict) and keys.MASK not in graph_globals:
        graph_globals[keys.MASK] = np.zeros(num_graphs, dtype=bool)

    return jraph.GraphsTuple(
        # Push all the nodes and edges to the final graph which is typically the padding graph
        n_node=np.array((num_graphs - 1) * [0] + [num_nodes]),
        n_edge=np.array((num_graphs - 1) * [0] + [num_edges]),
        nodes=jax.tree.map(np.zeros_like, graph.nodes),
        edges=jax.tree.map(np.zeros_like, graph.edges),
        globals=graph_globals,
        senders=jax.tree.map(np.zeros_like, graph.senders),
        receivers=jax.tree.map(np.zeros_like, graph.receivers),
    )


def stack_graphs_tuple(graph_list: list[jraph.GraphsTuple], np_=None) -> jraph.GraphsTuple:
    """Stacks a list of GraphsTuples with array or PyTree fields (e.g. dicts) into one batched
    GraphsTuple.
    """
    if np_ is None:
        np_ = tensorial_utils.infer_backend(graph_list)

    # Use jax map to stack PyTree structures across the batch
    stacked_nodes = jax.tree.map(lambda *args: np_.stack(args), *(g.nodes for g in graph_list))
    stacked_edges = jax.tree.map(lambda *args: np_.stack(args), *(g.edges for g in graph_list))
    stacked_globals = jax.tree.map(lambda *args: np_.stack(args), *(g.globals for g in graph_list))

    # Handle non-PyTree fields directly (these are just arrays or scalars)
    stacked_senders = np_.stack([g.senders for g in graph_list])
    stacked_receivers = np_.stack([g.receivers for g in graph_list])
    stacked_n_node = np_.stack([g.n_node for g in graph_list])
    stacked_n_edge = np_.stack([g.n_edge for g in graph_list])

    return jraph.GraphsTuple(
        nodes=stacked_nodes,
        edges=stacked_edges,
        globals=stacked_globals,
        senders=stacked_senders,
        receivers=stacked_receivers,
        n_node=stacked_n_node,
        n_edge=stacked_n_edge,
    )
