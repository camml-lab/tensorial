from functools import partial
import http.server
import threading
from unittest.mock import patch

import jraph
import pytest

# Do not import ase here to avoid circular imports during collection
from tensorial.gcnn.data._ase import (
    AseDataFetcher,
    AseDataFetchers,
    AseDataLoader,
    AseGraphs,
    ase_graph_module_from_datasets,
    ase_graph_module_kfold,
    ase_graph_module_random_split,
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


# ---- URL support -----------------------------------------------------------


class _CountingHandler(http.server.SimpleHTTPRequestHandler):
    """Serves a fixed directory and appends each GET to ``self.server.hits``."""

    def do_GET(self):
        self.server.hits.append(1)
        super().do_GET()

    def log_message(self, *args):
        pass  # keep test output clean


@pytest.fixture
def http_server(tmp_path):
    """Serve ``tmp_path`` over HTTP and yield ``(local_path, url, hits)``.

    A 10-structure water.xyz is written into the served directory so the same
    file can be referenced both locally and via its URL, and ``hits`` accumulates
    the number of GET requests the server has received.
    """
    import ase.build
    import ase.io

    file_path = tmp_path / "water.xyz"
    ase.io.write(str(file_path), [ase.build.molecule("H2O")] * 10)

    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(_CountingHandler, directory=str(tmp_path))
    )
    server.hits: list = []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield str(file_path), f"http://127.0.0.1:{server.server_port}/water.xyz", server.hits
    finally:
        server.shutdown()


def test_fetcher_downloads_url_and_caches(http_server, tmp_path):
    local_path, url, hits = http_server
    cache_dir = tmp_path / "cache"

    fetcher = AseDataFetcher(url, as_graphs={"r_max": 5.0}, cache_dir=cache_dir)
    graphs = fetcher.fetch()

    assert isinstance(graphs, AseGraphs)
    assert len(graphs) == 10
    assert int(graphs[0].n_node[-1]) == 3
    # the file was fetched from the URL and stored in the cache
    assert list(cache_dir.glob("*.xyz")), "expected a cached download"
    assert len(hits) == 1


def test_fetcher_reuses_cache_on_second_fetch(http_server, tmp_path):
    _, url, hits = http_server
    cache_dir = tmp_path / "cache"

    fetcher = AseDataFetcher(url, as_graphs={"r_max": 5.0}, cache_dir=cache_dir)
    fetcher.fetch()
    first_hits = len(hits)

    fetcher.fetch()

    # the cached file is served on the second fetch, so no new download occurs
    assert len(hits) == first_hits


def test_fetcher_local_path_uses_no_cache(http_server, tmp_path):
    local_path, url, hits = http_server
    cache_dir = tmp_path / "cache"

    fetcher = AseDataFetcher(local_path, as_graphs={"r_max": 5.0}, cache_dir=cache_dir)
    graphs = fetcher.fetch()

    assert isinstance(graphs, AseGraphs)
    assert len(graphs) == 10
    # local files are read directly: nothing downloaded, nothing cached
    assert len(hits) == 0
    assert not cache_dir.exists()


def test_fetchers_mixes_url_local_and_none(http_server, tmp_path):
    local_path, url, hits = http_server
    cache_dir = tmp_path / "cache"

    fetchers = AseDataFetchers(
        {"train": url, "val": None, "test": local_path},
        as_graphs={"r_max": 5.0},
        cache_dir=cache_dir,
    )
    fetched = fetchers.fetch()

    assert set(fetched) == {"train", "test"}
    assert all(isinstance(v, AseGraphs) for v in fetched.values())
    assert all(len(v) == 10 for v in fetched.values())
    # only the URL split is downloaded into the cache
    assert len(hits) == 1
    assert list(cache_dir.glob("*.xyz")), "expected a cached download for the URL split"
    # the local split is read from disk, not downloaded
    assert len(fetched["test"]) == 10


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
