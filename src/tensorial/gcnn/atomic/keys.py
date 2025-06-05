from typing import Final

__all__ = (
    "ENERGY_PER_ATOM",
    "TOTAL_ENERGY",
    "FORCES",
    "STRESS",
    "VIRIAL",
    "PBC",
    "ATOMIC_NUMBERS",
    "ASE_GLOBAL_KEYS",
    "ASE_ATOM_KEYS",
)

ENERGY_PER_ATOM: Final[str] = "energy/atom"
TOTAL_ENERGY: Final[str] = "energy"
ENERGY: Final[str] = "energy"
FORCES: Final[str] = "forces"
STRESS: Final[str] = "stress"
VIRIAL: Final[str] = "virial"
PBC: Final[str] = "pbc"
ATOMIC_NUMBERS: Final[str] = "atomic_numbers"

# Global quantities
ASE_GLOBAL_KEYS: Final[set[str]] = {"energy", "free_energy", "stress", "magmom"}
# Per-atom quantities
ASE_ATOM_KEYS: Final[set[str]] = {"numbers", "forces", "stresses", "charges", "magmoms", "energies"}
