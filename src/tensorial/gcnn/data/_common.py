import collections
import enum
from typing import Any

import jraph
import reax

__all__ = "GraphBatch", "GraphDataset", "GraphPadding", "GraphAttributes", "BatchMode"

GraphsOrGraphsTuple = jraph.GraphsTuple | tuple[jraph.GraphsTuple, ...]


class GraphBatch(tuple):
    inputs: jraph.GraphsTuple
    targets: Any | None


GraphDataset = reax.data.Dataset[GraphBatch]  # pylint: disable=invalid-name
GraphPadding = collections.namedtuple("GraphPadding", ["n_nodes", "n_edges", "n_graphs"])


class GraphAttributes(enum.IntFlag):
    NODES = 0b0001
    EDGES = 0b0010
    GLOBALS = 0b0100
    ALL = NODES | EDGES | GLOBALS


class BatchMode(str, enum.Enum):
    IMPLICIT = "implicit"
    EXPLICIT = "explicit"
