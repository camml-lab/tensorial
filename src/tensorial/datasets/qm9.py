import collections.abc
import io
import logging
import os
import pathlib
import tarfile
import types
from typing import Any, Final, Optional, TypedDict, Union
import urllib.request

import ase.data
import jraph
import numpy as np
import tqdm

from .. import base, gcnn

__all__ = ("Qm9",)

MoleculeDict = dict[str, Any]

_LOGGER = logging.getLogger(__name__)

QM9_XYZ_LABELS: Final[list[str]] = [
    "tag",
    "index",
    "A",
    "B",
    "C",
    "mu",
    "alpha",
    "homo",
    "lumo",
    "gap",
    "r2",
    "zpve",
    "U0",
    "U",
    "H",
    "G",
    "Cv",
]


class GraphOptions(TypedDict):
    r_max: float
    self_edges: bool
    node_attrs: list[Union[str, tuple[str, str]]]
    graph_attrs: list[Union[str, tuple[str, str]]]
    np_: types.ModuleType


class Qm9(collections.abc.Sequence):
    URL: Final[str] = "https://springernature.figshare.com/ndownloader/files/3195389"
    FILENAME: Final[str] = "dsgdb9nsd.xyz.tar.bz2"
    QM9_STRUCTURES: Final[str] = "qm9_structures"

    def __init__(
        self,
        data_dir: str = "data/",
        download: bool = True,
        limit: Optional[int] = None,
        as_graphs: Optional[dict] = None,
    ):
        # Params
        self._data_dir: Final[str] = data_dir
        self._download: Final[bool] = download
        self._to_graphs: Final[dict] = as_graphs

        # State
        if download:
            self._do_download("/".join([self.URL, self.FILENAME]), self.FILENAME)

        archive_path = pathlib.Path(self._data_dir) / self.FILENAME
        self._data = self._extract_tarball(archive_path, limit)

    def __getitem__(self, item):
        entry = self._data[item]
        if self._to_graphs and not isinstance(entry, jraph.GraphsTuple):
            # Lazily convert the first time
            entry = self.to_graph(entry)
            self._data[item] = entry

        return entry

    def __len__(self) -> int:
        return len(self._data)

    def _do_download(self, url: str, filename: str):
        """Download the file at the URL to our data dir."""
        if not os.path.exists(self._data_dir):
            os.makedirs(self._data_dir)

        out_file = os.path.join(self._data_dir, filename)
        if not os.path.isfile(out_file):
            urllib.request.urlretrieve(url, out_file)  # nosec
            _LOGGER.info("downloaded %s to %s", url, self._data_dir)

    def _extract_tarball(self, archive_path, limit=None) -> list[MoleculeDict]:
        molecules = []
        with tarfile.open(archive_path) as file:
            members = file.getmembers()
            if limit:
                members = members[:limit]
            for entry in tqdm.tqdm(members):
                file_handle = file.extractfile(entry.name)
                out = read_qm9(io.TextIOWrapper(file_handle, encoding="utf-8"))
                out["filename"] = entry.name
                molecules.append(out)

        return molecules

    def to_graph(self, entry: MoleculeDict) -> jraph.GraphsTuple:
        return to_graph(entry, **self._to_graphs)


def _do_extract(archive_path, entry) -> MoleculeDict:
    with tarfile.open(archive_path) as file:
        file_handle = file.extractfile(entry.name)
        out = read_qm9(io.TextIOWrapper(file_handle, encoding="utf-8"))
        out["filename"] = entry.name
        return out


def read_qm9(file_handle) -> MoleculeDict:
    """
    Format description can be found here:
    https://springernature.figshare.com/articles/dataset/Readme_file_Data_description_for_Quantum_chemistry_structures_and_properties_of_134_kilo_molecules_/1057641?backTo=%2Fcollections%2FQuantum_chemistry_structures_and_properties_of_134_kilo_molecules%2F978904&file=3195392
    """
    if isinstance(file_handle, io.BytesIO):
        file_handle = io.TextIOWrapper(file_handle, encoding="utf-8")

    lines = file_handle.readlines()
    num_atoms = int(lines[0].strip())
    properties = lines[1].split()  # Contains properties like energy, dipole moment, etc.
    atoms = [line.split(maxsplit=1) for line in lines[2 : 2 + num_atoms]]

    # Parse atomic symbols and positions
    species = [atom[0] for atom in atoms]
    coords = np.stack(
        [np.fromstring(atom[1].replace("*^", "E"), dtype=float, sep=" ")[:3] for atom in atoms]
    )

    entries = {
        "positions": coords,
        "species": species,
    }

    # Now add the properties
    for i, (label, prop) in enumerate(zip(QM9_XYZ_LABELS, properties)):
        if i == 1:
            prop = int(prop)
        elif i > 1:
            prop = float(prop)

        entries[label] = prop

    return entries


def to_graph(
    entry: MoleculeDict,
    r_max: float,
    self_edges: bool = False,
    node_attrs: list[Union[str, tuple[str, str]]] = None,
    graph_attrs: list[Union[str, tuple[str, str]]] = None,
    np_=np,
) -> jraph.GraphsTuple:
    n_nodes = len(entry["species"])

    # Convert species labels to numbers
    atomic_numbers = np.fromiter(
        map(lambda symbol: ase.data.atomic_numbers[symbol], entry["species"]), dtype=float
    )
    node_attrs_ = {gcnn.atomic.ATOMIC_NUMBERS: atomic_numbers}
    if node_attrs:
        for key in node_attrs:
            if isinstance(key, str):
                label = key
            elif isinstance(key, tuple):
                key, label = key
            else:
                raise ValueError(f"Not attributes key must be str or tuple, got {type(key)}")

            attr = base.atleast_1d(entry[key], np_=np_)
            if attr.shape[0] != n_nodes:
                attr = attr.reshape(n_nodes, attr.shape[0])

            node_attrs_[label] = attr

    graph_attrs_ = {}
    if graph_attrs:
        for key in graph_attrs:
            if isinstance(key, str):
                label = key
            elif isinstance(key, tuple):
                key, label = key
            else:
                raise ValueError(f"Not attributes key must be str or tuple, got {type(key)}")

            attr = base.atleast_1d(entry[key], np_=np_)

            graph_attrs_[label] = attr

    return gcnn.graph_from_points(
        entry["positions"],
        r_max,
        cell=None,
        fractional_positions=False,
        self_interaction=False,
        strict_self_interaction=self_edges,
        pbc=None,
        nodes=node_attrs_,
        graph_globals=graph_attrs_,
    )
