"""The revised MD17 (rMD17) benchmark dataset."""

import logging
import os
import pathlib
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from .. import data, utils
from ..gcnn.data import _ase, _datamodule
from ..gcnn.data import _urls as urls

if TYPE_CHECKING:
    import ase

__all__ = ("Rmd17", "rmd17_graph_module")

_LOGGER = logging.getLogger(__name__)

# 1 kcal/mol in eV (CODATA 2014, the value ``ase.units`` uses)
KCAL_PER_MOL_TO_EV: Final[float] = 0.0433641043

# figshare file ids for the individual files of https://doi.org/10.6084/m9.figshare.12672038
_FIGSHARE_URL: Final[str] = "https://ndownloader.figshare.com/files/{}"
_MOLECULE_FILES: Final[dict[str, int]] = {
    "aspirin": 62265757,
    "azobenzene": 62265754,
    "benzene": 62265739,
    "ethanol": 62265733,
    "malonaldehyde": 62265736,
    "naphthalene": 62265751,
    "paracetamol": 62265760,
    "salicylic": 62265748,
    "toluene": 62265742,
    "uracil": 62265745,
}
_SPLIT_FILES: Final[dict[str, int]] = {
    "index_train_01.csv": 62265793,
    "index_train_02.csv": 62265772,
    "index_train_03.csv": 62265775,
    "index_train_04.csv": 62265766,
    "index_train_05.csv": 62265778,
    "index_test_01.csv": 62265781,
    "index_test_02.csv": 62265769,
    "index_test_03.csv": 62265787,
    "index_test_04.csv": 62265790,
    "index_test_05.csv": 62265784,
}


class Rmd17(data.DataFetcher[_datamodule.PreSplitDataset]):
    """The revised MD17 dataset (Christensen & von Lilienfeld, 2020).

    Ten small organic molecules, each with 100,000 conformations from molecular
    dynamics, recomputed at the PBE/def2-SVP level.  The dataset comes with five
    standard splits of 1000 training and 1000 test conformations; the dataset
    authors advise against training on more than 1000 conformations, as the frames
    are correlated.

    Files are downloaded from figshare to ``data_dir`` on first use.  Energies and
    forces are converted from kcal/mol to eV.

    :meth:`fetch` returns the ``train`` and ``test`` sets of the chosen split (there
    is no standard validation set; use ``val_from_train`` on the data module to hold
    some of the training set out).

    Args:
        molecule: one of :attr:`MOLECULES`
        split: which of the five standard splits to use (1 to 5)
        as_graphs: keyword arguments for
            :func:`~tensorial.gcnn.atomic.graph_from_ase`, e.g. ``{"r_max": 5.0}``
        data_dir: directory in which to store the downloaded files
        download: download missing files.  Set to ``False`` to use only files
            already in ``data_dir``.
        limit: if set, use only the first ``limit`` conformations of the train and
            test sets (e.g. for quick tests)

    Example:
        >>> train_test = Rmd17("aspirin", as_graphs={"r_max": 5.0}).fetch()
        >>> len(train_test["train"]), len(train_test["test"])
        (1000, 1000)
    """

    MOLECULES: Final[tuple[str, ...]] = tuple(_MOLECULE_FILES)
    N_SPLITS: Final[int] = 5

    def __init__(
        self,
        molecule: str,
        split: int = 1,
        as_graphs: dict[str, Any] | None = None,
        data_dir: str | os.PathLike = "data/",
        download: bool = True,
        limit: int | None = None,
    ):
        if molecule not in _MOLECULE_FILES:
            raise ValueError(
                f"Unknown rMD17 molecule '{molecule}', expected one of {self.MOLECULES}"
            )
        if not 1 <= split <= self.N_SPLITS:
            raise ValueError(f"split must be between 1 and {self.N_SPLITS}, got {split}")

        # Params
        self._molecule: Final[str] = molecule
        self._split: Final[int] = split
        self._as_graphs: Final[dict[str, Any] | None] = as_graphs
        self._data_dir: Final[pathlib.Path] = pathlib.Path(data_dir) / "rmd17"
        self._download: Final[bool] = download
        self._limit: Final[int | None] = limit

    def fetch(self) -> _datamodule.PreSplitDataset:
        """Return the ``train`` and ``test`` sets of the chosen split."""
        npz = np.load(
            self._get_file(f"rmd17_{self._molecule}.npz", _MOLECULE_FILES[self._molecule])
        )
        train_idx = self._read_indices("train")
        test_idx = self._read_indices("test")

        return _datamodule.PreSplitDataset(
            train=self._make_sequence(npz, train_idx),
            test=self._make_sequence(npz, test_idx),
        )

    def _read_indices(self, which: str) -> np.ndarray:
        filename = f"index_{which}_{self._split:02d}.csv"
        indices = np.loadtxt(self._get_file(filename, _SPLIT_FILES[filename]), dtype=int)
        return indices[: self._limit]

    def _get_file(self, filename: str, file_id: int) -> pathlib.Path:
        path = self._data_dir / filename
        if not path.exists():
            if not self._download:
                raise FileNotFoundError(f"rMD17 file not found and download=False: {path}")
            self._data_dir.mkdir(parents=True, exist_ok=True)
            url = _FIGSHARE_URL.format(file_id)
            _LOGGER.info("downloading %s from %s", filename, url)
            urls._download(url, path)  # noqa: SLF001
        return path

    def _make_sequence(self, npz, indices: np.ndarray):
        structures = to_ase(npz, indices)
        if self._as_graphs is None:
            return structures
        return _ase.AseGraphs(structures, as_graphs=self._as_graphs)


def to_ase(npz, indices: np.ndarray | None = None) -> "list[ase.Atoms]":
    """Convert (a subset of) an rMD17 ``.npz`` file to :class:`ase.Atoms` in eV units.

    Args:
        npz: the loaded ``.npz`` file, or any mapping with the rMD17 keys
        indices: the conformations to convert, or ``None`` for all of them

    Returns:
        a list of molecules, each with a single-point calculator holding its energy
        and forces
    """
    ase = utils.optional_import("ase")
    singlepoint = utils.optional_import("ase.calculators.singlepoint")

    numbers = npz["nuclear_charges"]
    coords = npz["coords"]
    energies = npz["energies"]
    forces = npz["forces"]
    if indices is None:
        indices = np.arange(len(energies))

    structures = []
    for idx in indices:
        atoms = ase.Atoms(numbers=numbers, positions=coords[idx], pbc=False)
        atoms.calc = singlepoint.SinglePointCalculator(
            atoms,
            energy=float(energies[idx]) * KCAL_PER_MOL_TO_EV,
            forces=forces[idx] * KCAL_PER_MOL_TO_EV,
        )
        structures.append(atoms)

    return structures


def rmd17_graph_module(
    molecule: str,
    as_graphs: dict[str, Any],
    *,
    split: int = 1,
    val_from_train: int | float | None = 50,
    data_dir: str | os.PathLike = "data/",
    limit: int | None = None,
    batch_size: int = 32,
    batch_mode: str = "implicit",
    pad_to_multiple=None,
) -> "_datamodule.GraphDataModule":
    """Build a :class:`~tensorial.gcnn.data.GraphDataModule` for an rMD17 molecule.

    By default, uses the standard 950/50 train/validation split of the 1000
    training conformations, and the 1000 conformations of the matching test split.

    Args:
        molecule: one of :attr:`Rmd17.MOLECULES`
        as_graphs: keyword arguments for
            :func:`~tensorial.gcnn.atomic.graph_from_ase`, e.g. ``{"r_max": 5.0}``
        split: which of the five standard splits to use (1 to 5)
        val_from_train: how many training conformations to hold out for validation
        data_dir: directory in which to store the downloaded files
        limit: if set, use only the first ``limit`` train and test conformations
    """
    return _datamodule.GraphDataModule.from_datasets(
        Rmd17(molecule, split=split, as_graphs=as_graphs, data_dir=data_dir, limit=limit),
        val_from_train=val_from_train,
        batch_size=batch_size,
        batch_mode=batch_mode,
        pad_to_multiple=pad_to_multiple,
    )
