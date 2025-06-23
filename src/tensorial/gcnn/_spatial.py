import logging
import numbers
from typing import TYPE_CHECKING, Final, Optional

import e3nn_jax as e3j
import jax
import jax.numpy as jnp
import jaxtyping as jt
import jraph
import numpy as np
import reax.metrics

from . import keys
from .. import base, geometry, nn_utils

if TYPE_CHECKING:
    import tensorial.typing as tt

_LOGGER = logging.getLogger(__name__)

__all__ = ("graph_from_points", "with_edge_vectors")


# @jt.jaxtyped(typechecker=beartype.beartype)
def graph_from_points(
    pos: "jt.Float[tt.ArrayType, 'n_nodes 3']",
    r_max: numbers.Number,
    *,
    fractional_positions: bool = False,
    self_interaction: bool = True,
    strict_self_interaction: bool = False,
    cell: "Optional[tt.CellType]" = None,
    pbc: "Optional[bool | tt.PbcType]" = None,
    nodes: Optional[dict[str, jt.Num[jax.typing.ArrayLike, "n_nodes *"]]] = None,
    edges: dict = None,
    graph_globals: dict = None,
    np_=np,
) -> jraph.GraphsTuple:
    """
    Create a jraph Graph from a set of atomic positions and other related data.

    :param pos: a [N, 3] array of atomic positions
    :param r_max: the cutoff radius to use for identifying neighbours
    :param fractional_positions: if ``True``, `pos` are interpreted as fractional positions
    :param self_interaction: if ``True``, edges are created between an atom and itself in other unit
        cells
    :param strict_self_interaction:  if ``True``, edges are created between an atom and itself
        within the central unit cell
    :param cell: a [3, 3] array of unit cell vectors (in row-major format)
    :param pbc: a ``bool`` of a sequence of three `bool`s indicating whether the space is periodic
        in x, y, z directions
    :param nodes: a dictionary containing additional data relating to each node, it should contain
        arrays of shape [N, ...]
    :param graph_globals: a dictionary containing additional global data
    :return: the corresponding jraph Graph
    """
    pos = np_.asarray(pos)
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
        if cell is not None:
            raise ValueError("A cell was provided without PBCs")
        pbc = False

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

    neighbour_finder = geometry.neighbour_finder(
        r_max,
        cell,
        pbc=pbc,
        include_self=strict_self_interaction,
        include_images=self_interaction,
    )
    neighbour_list = neighbour_finder.get_neighbours(pos)
    from_idx, to_idx, cell_shifts = tuple(map(np_.array, neighbour_list.get_edges()))

    nodes[keys.POSITIONS] = pos

    graph_globals = graph_globals or {}
    if pbc is not None:
        graph_globals[keys.PBC] = np_.array(pbc, dtype=bool)
    if cell is not None:
        graph_globals[keys.CELL] = cell
    # We have to pad out the globals to make things like batching work
    graph_globals = {
        key: np_.expand_dims(base.atleast_1d(value), 0)
        for key, value in graph_globals.items()
        if value is not None
    }

    edges = edges or {}
    edges = {key: value[from_idx, to_idx] for key, value in edges.items()}
    # Make sure the edge arrays have array-like (rather than scalar) entries for each edge
    edges = {
        key: np_.expand_dims(value, -1) if value.ndim == 1 else value
        for key, value in edges.items()
    }
    if cell is not None:
        edges[keys.EDGE_CELL_SHIFTS] = cell_shifts

    return jraph.GraphsTuple(
        nodes=nodes,
        edges=edges,
        senders=from_idx,
        receivers=to_idx,
        globals=graph_globals,
        n_node=np_.array([len(pos)]),
        n_edge=np_.array([len(from_idx)]),
    )


def with_edge_vectors(
    graph: jraph.GraphsTuple,
    with_lengths: bool = True,
    as_irreps_array: Optional[bool] = True,
) -> jraph.GraphsTuple:
    """Compute edge displacements for edge vectors in a graph.

    This will add edge attributes corresponding that cache the vectors and displacements, meaning
    that they will not be recalculated if already done so.
    """
    edges = graph.edges
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

    edge_mask = graph.edges.get(keys.MASK)
    if edge_mask is not None:
        edge_mask = reax.metrics.utils.prepare_mask(edge_vecs, edge_mask)
        edge_vecs = jnp.where(edge_mask, edge_vecs, 1.0)

    if not isinstance(edge_vecs, e3j.IrrepsArray) and as_irreps_array:
        edge_vecs = e3j.IrrepsArray("1o", edge_vecs)
    edges[keys.EDGE_VECTORS] = edge_vecs

    # To allow grad to work, we need to mask off the padded edge-vectors that are zero, see:
    # * https://github.com/google/jax/issues/6484,
    # * https://stackoverflow.com/q/74864427/1257417
    if with_lengths:
        lengths = jnp.expand_dims(jnp.linalg.norm(base.as_array(edge_vecs), axis=-1), -1)
        if edge_mask is not None:
            lengths = jnp.where(edge_mask, lengths, 0.0)
        if as_irreps_array:
            lengths = e3j.as_irreps_array(lengths)

        edges[keys.EDGE_LENGTHS] = lengths

    graph = graph._replace(edges=edges)
    return graph


def _pairwise_sq_distances(pos: "tt.ArrayType", mask: "Optional[tt.ArrayType]") -> "tt.ArrayType":
    # x_shape: (num_nodes_in_graph, 3)
    if mask is not None:
        mask = nn_utils.prepare_mask(mask, pos)
        pos = jnp.where(mask, pos, jnp.inf)

    diff = pos[:, None, :] - pos[None, :, :]
    return (diff**2).sum(axis=2)


def _update_positions(
    padded_graph: jraph.GraphsTuple, new_pos: "tt.ArrayType", r_max: float
) -> jraph.GraphsTuple:
    nodes_dict = padded_graph.nodes
    if not nodes_dict[keys.POSITIONS].shape == new_pos.shape:
        raise ValueError(
            f"New positions ({new_pos.shape}) and current position arrays "
            f"({nodes_dict[keys.POSITIONS].shape}) do not have the same shape"
        )

    nodes_dict["positions"] = new_pos

    dists_mtx = _pairwise_sq_distances(new_pos, nodes_dict[keys.MASK])
    r_max_sq: Final = r_max**2
    pairs = jnp.argwhere(
        (dists_mtx < r_max_sq) & (dists_mtx > 0.0),
        size=padded_graph.senders.shape[0],
        fill_value=-1,  # Use -1 to identify the padding values
    )

    n_edge: Final = padded_graph.edges[keys.MASK].shape[0]
    new_edge_mask = pairs[:, 0] >= 0  # i.e. not -1
    new_n_edge = sum(new_edge_mask)
    new_senders = jnp.where(new_edge_mask, pairs[:, 0], 0.0)
    new_receivers = jnp.where(new_edge_mask, pairs[:, 1], 0.0)

    return padded_graph._replace(
        senders=new_senders,
        receivers=new_receivers,
        n_edge=jnp.array([new_n_edge, n_edge - new_n_edge]),
        edges={"mask": new_edge_mask},
        nodes=nodes_dict,
    )
