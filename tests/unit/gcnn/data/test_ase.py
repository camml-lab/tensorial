from unittest.mock import patch

import jraph
import pytest

# Do not import ase here to avoid circular imports during collection
from tensorial.gcnn.data._ase import (
    AseDataFetcher,
    AseDataFetchers,
    AseDataLoader,
    AseGraphs,
    ase_graph_module_random_split,
    ase_graph_module_from_datasets,
    ase_graph_module_kfold,
)


@pytest.fixture
def temp_xyz_file(tmp_path):
    import ase.build
    import ase.io

    # Create structures
    atoms = [ase.build.molecule("H2O"), ase.build.molecule("H2O")]

    # Save to temp file
    file_path = tmp_path / "test.xyz"
    ase.io.write(str(file_path), atoms)
    return str(file_path)


@pytest.fixture
def mock_atomic_conversion():
    with patch("tensorial.gcnn.data._ase.atomic.graph_from_ase") as mock_conv:
        # Return a dummy GraphsTuple
        mock_conv.return_value = jraph.GraphsTuple(
            n_node=None,
            n_edge=None,
            nodes=None,
            edges=None,
            globals=None,
            senders=None,
            receivers=None,
        )
        yield mock_conv


def test_ase_data_loader_loading(temp_xyz_file):
    # Verify it loads the correct number of items
    loader = AseDataLoader(path=temp_xyz_file)
    assert len(loader) == 2
    assert loader[0].get_chemical_formula() == "H2O"


def test_ase_data_loader_lazy_conversion(temp_xyz_file, mock_atomic_conversion):
    # Pass as_graphs to trigger lazy conversion
    # Note: AseDataLoader(..., as_graphs=...) triggers __getitem__(0) in __init__
    # to validate the arguments.

    loader = AseDataLoader(path=temp_xyz_file, as_graphs={"r_max": 3.0})

    # Due to the validation in __init__, conversion of [0] happens during init
    assert mock_atomic_conversion.call_count == 1

    # Access item 1 - should trigger conversion
    item = loader[1]

    assert isinstance(item, jraph.GraphsTuple)
    assert mock_atomic_conversion.call_count == 2


# ---- AseGraphs / AseDataFetcher / AseDataFetchers -------------------------


@pytest.fixture
def water_dataset_file(tmp_path):
    """A temp file with 10 H2O structures, enough to drive a full GraphDataModule."""
    import ase.build
    import ase.io

    atoms = [ase.build.molecule("H2O")] * 10
    file_path = tmp_path / "water.xyz"
    ase.io.write(str(file_path), atoms)
    return str(file_path)


class _Stage:
    engine = None


def _build_and_setup(module):
    module.prepare_data()
    module.setup(_Stage())


def test_ase_data_fetcher_returns_ase_graphs(water_dataset_file):
    fetcher = AseDataFetcher(water_dataset_file, as_graphs={"r_max": 5.0})
    graphs = fetcher.fetch()

    assert isinstance(graphs, AseGraphs)
    assert len(graphs) == 10
    assert isinstance(graphs[0], jraph.GraphsTuple)
    # H2O has 3 atoms
    assert int(graphs[0].n_node[-1]) == 3


def test_ase_data_fetcher_limit(water_dataset_file):
    fetcher = AseDataFetcher(water_dataset_file, as_graphs={"r_max": 5.0}, limit=4)
    graphs = fetcher.fetch()

    assert len(graphs) == 4


def test_ase_data_fetchers_returns_mapping(water_dataset_file):
    fetchers = AseDataFetchers(
        {"train": water_dataset_file, "test": water_dataset_file}, as_graphs={"r_max": 5.0}
    )
    fetched = fetchers.fetch()

    assert set(fetched) == {"train", "test"}
    assert all(isinstance(v, AseGraphs) for v in fetched.values())
    assert all(len(v) == 10 for v in fetched.values())


def test_ase_data_fetchers_skips_none_paths(water_dataset_file):
    # A split the user did not provide is a ``None`` value and must be skipped,
    # not raised on -- ``PreSplitDataset`` is a ``total=False`` mapping.
    fetchers = AseDataFetchers(
        {"train": water_dataset_file, "val": None, "test": water_dataset_file},
        as_graphs={"r_max": 5.0},
    )
    fetched = fetchers.fetch()

    assert set(fetched) == {"train", "test"}


# ---- GraphDataModule builders ---------------------------------------------


def test_random_split_from_ase_builds_module(water_dataset_file):
    module = ase_graph_module_random_split(
        water_dataset_file,
        as_graphs={"r_max": 5.0},
        train_val_test_split=(0.6, 0.2, 0.2),
        batch_size=2,
    )
    _build_and_setup(module)

    # 10 structures -> 6/2/2
    assert len(module.data_train) == 6
    assert len(module.data_val) == 2
    assert len(module.data_test) == 2
    assert len(module.train_dataloader()) > 0


def test_kfold_from_ase_builds_module(water_dataset_file):
    module = ase_graph_module_kfold(
        water_dataset_file,
        as_graphs={"r_max": 5.0},
        fold=0,
        n_folds=4,
        test_fraction=0.2,
        batch_size=2,
    )
    _build_and_setup(module)

    # test = 20% = 2; remaining 8 -> 2 val (fold of 8/4) + 6 train
    assert len(module.data_test) == 2
    assert len(module.data_val) == 2
    assert len(module.data_train) == 6


def test_from_datasets_ase_partial_splits(water_dataset_file):
    module = ase_graph_module_from_datasets(
        {"r_max": 5.0}, train=water_dataset_file, batch_size=2
    )
    _build_and_setup(module)

    assert len(module.data_train) == 10
    assert module.data_val is None
    assert module.data_test is None


def test_from_datasets_ase_all_none_rejected_at_setup(water_dataset_file):
    import reax

    module = ase_graph_module_from_datasets({"r_max": 5.0}, batch_size=2)
    module.prepare_data()
    with pytest.raises(reax.exceptions.MisconfigurationException, match="all None datasets"):
        module.setup(_Stage())
