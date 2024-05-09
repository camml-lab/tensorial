# -*- coding: utf-8 -*-
import logging
from typing import Dict, Optional, Tuple, Union

import e3nn_jax as e3j
import equinox
import jax
import jax.numpy as jnp
import jraph

from tensorial import distances, nn_utils
import tensorial.gcnn.keys

from . import keys

_LOGGER = logging.getLogger(__name__)

__all__ = ("graph_from_points", "with_edge_vectors")


def graph_from_points(
    pos: jax.typing.ArrayLike,
    r_max: float,
    fractional_positions: bool = False,
    self_interaction: bool = False,
    strict_self_interaction: bool = True,
    cell: jax.typing.ArrayLike = None,
    pbc: Optional[Union[bool, Tuple[bool, bool, bool]]] = None,
    nodes: Dict = None,
    edges: Dict = None,
    graph_globals: Dict = None,
) -> jraph.GraphsTuple:
    """
    Create a jraph Graph from a set of atomic positions and other related data.

    :param pos: a [3, N] array of atomic positions
    :param r_max: the cutoff radius to use for identifying neighbours
    :param fractional_positions: if `True`, `pos` are interpreted as fractional positions
    :param self_interaction: if `True`, edges are created between an atom and itself in other unit
        cells
    :param strict_self_interaction:  if `True`, edges are created between an atom and itself within
        the central unit cell
    :param cell: a [3, 3] array of unit cell vectors (in row-major format)
    :param pbc: a `bool` of a sequence of three `bool`s indicating whether the space is periodic in
        x, y, z directions
    :param nodes: a dictionary containing additional data relating to each node, it should contain
        arrays of shape [N, ...]
    :param graph_globals: a dictionary containing additional global data
    :return: the corresponding jraph Graph
    """
    if not pos.shape[-1] == 3:
        raise ValueError(f"pos must have shape [N, 3], got {pos.shape}")
    pos = jnp.asarray(pos)
    nodes = nodes if nodes else {}
    num_nodes = len(pos)

    nodes = nodes or {}
    for name, value in nodes.items():
        if value.shape[0] != num_nodes:
            raise ValueError(
                f"node attributes should have shape [N, ...], got {value.shape[0]} != {num_nodes} "
                f"for {name}"
            )

    if pbc is None:
        # there are no PBC if cell and pbc are not provided
        pbc = False
        if cell is not None:
            raise ValueError("A cell was provided without PBCs")

    if isinstance(pbc, bool):
        pbc = (pbc,) * 3

    if len(pbc) != 3:
        raise ValueError(f"PBC must have length 3: {pbc}")

    if fractional_positions:
        if cell is None:
            raise ValueError("Unit cell must be provided if fractional_positions is True")
        # Use PBC to mask and only transform periodic dimensions from fractional (assuming
        # non-periodic coordinates are already Cartesian)
        pos[:, pbc] = (pos @ cell)[pbc]

    neighbour_finder = distances.neighbour_finder(
        r_max,
        cell,
        pbc=pbc,
        include_self=self_interaction,
        include_images=strict_self_interaction,
    )
    get_neighbours = equinox.filter_jit(neighbour_finder.get_neighbours)
    neighbour_list = get_neighbours(pos, max_neighbours=neighbour_finder.estimate_neighbours(pos))
    if neighbour_list.did_overflow:
        _LOGGER.info(
            "Neighbour list was too small (%i) for amount of actual neighbours (%i), "
            "recalculating.",
            neighbour_list.max_neighbours,
            neighbour_list.actual_max_neighbours,
        )
        neighbour_list = neighbour_list.reallocate(pos)

    from_idx, to_idx, cell_shifts = neighbour_list.get_edges()

    nodes[keys.POSITIONS] = pos

    graph_globals = graph_globals or {}
    if pbc is not None:
        graph_globals[keys.PBC] = jnp.array(pbc, dtype=bool)
    graph_globals[keys.CELL] = cell
    # We have to pad out the globals to make things like batching work
    graph_globals = {
        key: jnp.expand_dims(value, 0) for key, value in graph_globals.items() if value is not None
    }

    edges = edges or {}
    edges = {key: value[from_idx, to_idx] for key, value in edges.items()}
    # Make sure the edge arrays have array-like (rather than scalar) entries for each edge
    edges = {
        key: jnp.expand_dims(value, -1) if value.ndim == 1 else value
        for key, value in edges.items()
    }
    edges[keys.EDGE_CELL_SHIFTS] = cell_shifts

    return jraph.GraphsTuple(
        nodes=nodes,
        edges=edges,
        senders=from_idx,
        receivers=to_idx,
        globals=graph_globals,
        n_node=jnp.array([len(pos)]),
        n_edge=jnp.array([len(from_idx)]),
    )


def with_edge_vectors(graph: jraph.GraphsTuple, with_lengths: bool = True) -> jraph.GraphsTuple:
    """Compute edge displacements for edge vectors in a graph.

    This will add edge attributes corresponding that cache the vectors and displacements, meaning
    that they will not be recalculated if already done so.
    """
    edges = graph.edges
    if keys.EDGE_VECTORS in edges:
        return graph

    pos = graph.nodes[keys.POSITIONS]
    edge_vecs = pos[graph.receivers] - pos[graph.senders]

    if keys.CELL in graph.globals:
        cell = graph.globals[keys.CELL]
        cell_shifts = edges[keys.EDGE_CELL_SHIFTS]
        shift_vectors = jnp.einsum(
            "ni,nij->nj",
            cell_shifts,
            jnp.repeat(cell, graph.n_edge, axis=0, total_repeat_length=edge_vecs.shape[0]),
        )
        edge_vecs = edge_vecs + shift_vectors

    edge_mask = graph.edges.get(tensorial.gcnn.keys.MASK)
    if edge_mask is not None:
        edge_mask = nn_utils.prepare_mask(edge_mask, edge_vecs)
        edge_vecs = jnp.where(edge_mask, edge_vecs, 1.0)
    edges[keys.EDGE_VECTORS] = e3j.IrrepsArray("1o", edge_vecs)

    # To allow grad to work, we need to mask off the padded edge vectors that are zero, see:
    # * https://github.com/google/jax/issues/6484,
    # * https://stackoverflow.com/q/74864427/1257417
    if with_lengths and keys.EDGE_LENGTHS not in edges:
        lengths = jnp.expand_dims(jnp.linalg.norm(edge_vecs, axis=-1), -1)
        if edge_mask is not None:
            lengths = jnp.where(edge_mask, lengths, 0.0)

        edges[keys.EDGE_LENGTHS] = lengths

    graph = graph._replace(edges=edges)
    return graph
