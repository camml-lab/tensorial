# -*- coding: utf-8 -*-
from typing import Final

POSITIONS: Final[str] = 'positions'

FEATURES: Final[str] = 'features'
ATTRIBUTES: Final[str] = 'attributes'

RADIAL_EMBEDDINGS: Final[str] = 'radial_embeddings'

# [n_edge, 3] tensor of displacement vectors associated to edges
EDGE_VECTORS: Final[str] = 'edge_vectors'
# An [n_edge] tensor of the lengths of EDGE_VECTORS
EDGE_LENGTHS: Final[str] = 'edge_lengths'
# An [n_edge, 3] tensor containing
EDGE_CELL_SHIFTS: Final[str] = 'edge_cell_shifts'

# The unit cell matrix
CELL: Final[str] = 'cell'
# Periodic boundary conditions
PBC: Final[str] = 'pbc'
# A species (or type) integer
SPECIES: Final[str] = 'species'
