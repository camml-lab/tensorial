from collections.abc import Callable, Sequence
import functools
import io
import logging
import os
import pathlib
import tarfile
from typing import Any, Final
import urllib.request

import ase
import jraph
import numpy as np
import reax
from typing_extensions import override

from ... import gcnn

Dataset = Any

__all__ = ("Qm9DataModule",)

_LOGGER = logging.getLogger(__name__)


class Qm9DataModule(reax.DataModule):
    """`REAX DataModule` for the QM9 dataset.

    Description from the original authors:

    Computational de novo design of new drugs and materials requires rigorous and unbiased
    exploration of chemical compound space. However, large uncharted territories persist due to its
    size scaling combinatorially with molecular size. We report computed geometric, energetic,
    electronic, and thermodynamic properties for 134k stable small organic molecules made up of
    CHONF. These molecules correspond to the subset of all 133,885 species with up to nine heavy
    atoms (CONF) out of the GDB-17 chemical universe of 166 billion organic molecules. We report
    geometries minimal in energy, corresponding harmonic frequencies, dipole moments,
    polarizabilities, along with energies, enthalpies, and free energies of atomization. All
    properties were calculated at the B3LYP/6-31G(2df,p) level of quantum chemistry. Furthermore,
    for the predominant stoichiometry, C7H10O2, there are 6,095 constitutional isomers among the
    134k molecules. We report energies, enthalpies, and free energies of atomization at the more
    accurate G4MP2 level of theory for all of them. As such, this data set provides quantum
    chemical properties for a relevant, consistent, and comprehensive chemical space of small
    organic molecules. This database may serve the benchmarking of existing methods, development
    of new methods, such as hybrid quantum mechanics/machine learning, and systematic
    identification of structure-property relationships.


    https://springernature.figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904/5
    """

    URL: Final[str] = "https://springernature.figshare.com/ndownloader/files/3195389"
    FILENAME: Final[str] = "dsgdb9nsd.xyz.tar.bz2"
    QM9_STRUCTURES: Final[str] = "qm9_structures"

    data_train: list[jraph.GraphsTuple] | None = None
    data_val: list[jraph.GraphsTuple] | None = None
    data_test: list[jraph.GraphsTuple] | None = None
    _max_padding: gcnn.data.GraphPadding = None

    def __init__(
        self,
        r_max: float,
        data_dir: str = "data/",
        train_val_test_split: Sequence[int | float] = (0.85, 0.05, 0.1),
        batch_size: int = 64,
        download: bool = True,
    ) -> None:
        """Initialize a `Qm9DataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation, and test split.
        :param batch_size: The batch size. Defaults to `64`.
        """
        super().__init__()

        # Params
        self._rmax = r_max
        self._data_dir: Final[str] = data_dir
        self._train_val_test_split: Final[tuple[int | float, ...]] = tuple(train_val_test_split)
        self._batch_size: Final[int] = batch_size
        self._download: Final[bool] = download

        # State
        self.batch_size_per_device = batch_size
        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None

    @override
    def prepare_data(self) -> None:
        """Download data if needed. REAX ensures that `self.prepare_data()` is called only within a
        single process on CPU, so you can safely add your downloading logic within. In case of
        multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        if self._download:
            self._do_download("/".join([self.URL, self.FILENAME]), self.FILENAME)

    @override
    def setup(self, stage: "reax.Stage", /) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by REAX before `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`, so be careful not to execute things like random
        split twice! Also, it is called after `self.prepare_data()` and there is a barrier in
        between which ensures that all the processes proceed to `self.setup()` once the data is
        prepared and available for use.

        :param stage:
            The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            tarball = pathlib.Path(self._data_dir) / self.FILENAME
            structures_dir = self._extract_tarball(tarball)

            # Now create a split based on the filename
            train, val, test = reax.data.random_split(
                stage.rngs,
                dataset=list(structures_dir.glob("*.xyz")),
                lengths=self._train_val_test_split,
            )
            datasets = dict(train=train, val=val, test=test)

            to_graph: Callable[[ase.Atoms], jraph.GraphsTuple] = functools.partial(
                gcnn.atomic.graph_from_ase,
                r_max=self._rmax,
                pbc=False,
                global_include_keys=("U0",),
                key_mapping={"U0": "energy"},
            )

            graph_datasets: dict[str, list[jraph.GraphsTuple]] = {}
            paddings: list[gcnn.data.GraphPadding] = []
            for set_name, filenames in datasets.items():
                graphs = []
                for path in filenames:
                    with open(path, encoding="utf-8") as handle:
                        # Read as ASE Atoms object
                        structure: ase.Atoms = read_qm9(handle)
                        # Now convert to graph
                        graphs.append(to_graph(structure))

                graph_datasets[set_name] = graphs
                paddings.append(gcnn.data.GraphBatcher.calculate_padding(graphs, self._batch_size))

            self.data_train = graph_datasets["train"]
            self.data_val = graph_datasets["val"]
            self.data_test = graph_datasets["test"]

            # Calculate a padding that will work for all the datasets.
            self._max_padding = gcnn.data.max_padding(*paddings)

    @override
    def train_dataloader(self) -> reax.DataLoader:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        if self.data_train is None:
            raise reax.exceptions.MisconfigurationException(
                "Must call setup() before requesting the dataloader"
            )

        return gcnn.data.GraphLoader(
            self.data_train,
            batch_size=self._batch_size,
            padding=self._max_padding,
            pad=True,
        )

    @override
    def val_dataloader(self) -> reax.DataLoader:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        if self.data_val is None:
            raise reax.exceptions.MisconfigurationException(
                "Must call setup() before requesting the dataloader"
            )

        return gcnn.data.GraphLoader(
            self.data_val,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            padding=self._max_padding,
            pad=True,
        )

    @override
    def test_dataloader(self) -> reax.DataLoader:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        if self.data_test is None:
            raise reax.exceptions.MisconfigurationException(
                "Must call setup() before requesting the dataloader"
            )

        return gcnn.data.GraphLoader(
            self.data_test,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            padding=self._max_padding,
            pad=True,
        )

    def _do_download(self, url: str, filename: str):
        """Download the file at the URL to our data dir."""
        if not os.path.exists(self._data_dir):
            os.makedirs(self._data_dir)

        out_file = os.path.join(self._data_dir, filename)
        if not os.path.isfile(out_file):
            urllib.request.urlretrieve(url, out_file)  # nosec
            _LOGGER.info("downloaded %s to %s", url, self._data_dir)

    def _extract_tarball(self, tar_path):
        structures_dir = pathlib.Path(self._data_dir) / self.QM9_STRUCTURES

        # Extract the full tarball
        if not structures_dir.exists():
            structures_dir.mkdir(exist_ok=True)
            with tarfile.open(tar_path, "r", encoding="utf-8") as tar:
                tar.extractall(structures_dir)  # nosec

        return structures_dir


def read_qm9(file_handle):
    # Format description can be found here:
    # https://springernature.figshare.com/articles/dataset/Readme_file_Data_description_for_Quantum_chemistry_structures_and_properties_of_134_kilo_molecules_/1057641?backTo=%2Fcollections%2FQuantum_chemistry_structures_and_properties_of_134_kilo_molecules%2F978904&file=3195392
    labels = [
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
    if isinstance(file_handle, io.BytesIO):
        file_handle = io.TextIOWrapper(file_handle, encoding="utf-8")
    lines = file_handle.readlines()
    num_atoms = int(lines[0].strip())
    properties = lines[1].split()  # Contains properties like energy, dipole moment, etc.
    atoms = [line.split(maxsplit=1) for line in lines[2 : 2 + num_atoms]]

    # Parse atomic symbols and positions
    species = [atom[0] for atom in atoms]
    coords = [np.fromstring(atom[1].replace("*^", "E"), dtype=float, sep=" ")[:3] for atom in atoms]

    molecule = ase.Atoms(positions=coords, symbols=species, cell=None)
    # Now add the properties
    for i, (label, prop) in enumerate(zip(labels, properties)):
        if i == 1:
            prop = np.array(prop, dtype=int)
        elif i > 1:
            prop = np.array(prop, dtype=float)
        molecule.arrays[label] = prop

    return molecule
