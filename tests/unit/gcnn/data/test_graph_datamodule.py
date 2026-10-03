"""Tests for :class:`tensorial.gcnn.data.GraphDataModule`."""

import math

import jraph
import numpy as np
import pytest
import reax

from tensorial import gcnn
from tensorial.data import PassthroughFetcher
from tensorial.gcnn.data import GraphDataModule, PreSplitDataset
from tensorial.gcnn.data._datamodule import KFoldSplit


def _dset(n: int):
    """A fetcher over a trivial dataset, matching the factory-method convention."""
    return PassthroughFetcher(_trivial_graphs(n))


def _trivial_graphs(n: int) -> list[jraph.GraphsTuple]:
    """A list of *distinguishable*, non-empty graphs to seed a ``GraphDataModule``.

    Each graph is identical in shape but carries a distinct ``globals["g"]``
    value, so identity / membership comparisons over the underlying sequence
    stay meaningful.
    """
    graphs: list[jraph.GraphsTuple] = []
    for i in range(n):
        graphs.append(
            jraph.GraphsTuple(
                n_node=np.array([2]),
                n_edge=np.array([3]),
                nodes={"f": np.full((2, 1), float(i), dtype=np.float32)},
                edges={"f": np.full((3, 1), float(i), dtype=np.float32)},
                globals={"g": np.array([float(i)], dtype=np.float32)},
                senders=np.zeros(3, dtype=np.int32),
                receivers=np.zeros(3, dtype=np.int32),
            )
        )
    return graphs


class _MockStage:
    engine = None


# ---- Constructor -----------------------------------------------------------


def test_stores_params():
    dm = GraphDataModule.from_kfold(
        PassthroughFetcher(_trivial_graphs(20)),
        fold=1,
        n_folds=4,
        seed=1234,
        test_fraction=0.1,
        batch_size=7,
    )
    assert dm._strategy._test_fraction == 0.1
    assert dm.batch_size == 7
    assert dm._strategy._fold == 1
    assert dm._strategy._n_folds == 4
    assert dm._strategy._seed == 1234


def test_batch_size_property():
    dm = GraphDataModule.from_random_split(_dset(10), batch_size=13)
    assert dm.batch_size == 13


def test_batch_size_is_read_only():
    dm = GraphDataModule.from_random_split(_dset(10), batch_size=13)
    with pytest.raises(AttributeError):
        dm.batch_size = 99  # type: ignore[misc]


def test_string_batch_mode_coercion():
    for raw, enum in [
        ("implicit", gcnn.data.BatchMode.IMPLICIT),
        ("explicit", gcnn.data.BatchMode.EXPLICIT),
    ]:
        dm = GraphDataModule.from_random_split(_dset(10), batch_mode=raw)
        assert dm._batch_mode is enum


def test_invalid_batch_mode_rejected():
    with pytest.raises(ValueError):
        GraphDataModule.from_random_split(_dset(10), batch_mode="bogus")


def test_initial_state_has_no_datasets():
    dm = GraphDataModule.from_random_split(_dset(10))
    assert dm.data_train is None
    assert dm.data_val is None
    assert dm.data_test is None
    assert dm._max_padding is None


# ---- Pre-setup dataloader access ------------------------------------------


@pytest.mark.parametrize("method", ["train_dataloader", "val_dataloader", "test_dataloader"])
def test_dataloader_before_setup_raises(method):
    dm = GraphDataModule.from_random_split(_dset(10))
    with pytest.raises(reax.exceptions.MisconfigurationException, match="setup"):
        getattr(dm, method)()


# ---- Post-setup: dataset split and loaders --------------------------------


def _setup(dm: GraphDataModule) -> None:
    dm.prepare_data()
    dm.setup(_MockStage())


def test_setup_populates_datasets():
    dm = GraphDataModule.from_random_split(
        _dset(20), train_val_test_split=(0.8, 0.1, 0.1), batch_size=2
    )
    _setup(dm)

    assert dm.data_train is not None
    assert dm.data_val is not None
    assert dm.data_test is not None

    # Every original dataset element must land in exactly one split.  The
    # underlying dataset is a list of 20 distinct graphs, so the three
    # subsets must partition the original list by object identity.
    all_seen: list = []
    for split in (dm.data_train, dm.data_val, dm.data_test):
        all_seen.extend(split)
    assert len(all_seen) == 20
    # No duplicate (same object appears in two splits)
    assert len(set(map(id, all_seen))) == 20


def test_setup_computes_max_padding():
    dm = GraphDataModule.from_random_split(
        _dset(40), train_val_test_split=(0.8, 0.1, 0.1), batch_size=2
    )
    _setup(dm)
    assert dm._max_padding is not None
    # Implicit padding for 2-node graphs: n_nodes = 2*2 + 1 = 5,
    # n_edges = 3*2 = 6, n_graphs = batch_size + 1 = 3
    assert dm._max_padding.n_nodes == 5
    assert dm._max_padding.n_edges == 6
    assert dm._max_padding.n_graphs == 3


def test_loaders_return_graph_loader():
    dm = GraphDataModule.from_random_split(
        _dset(20), train_val_test_split=(0.8, 0.1, 0.1), batch_size=2
    )
    _setup(dm)
    for name in ("train_dataloader", "val_dataloader", "test_dataloader"):
        dl = getattr(dm, name)()
        assert isinstance(dl, gcnn.data.GraphLoader)


def test_train_loader_uses_constructor_batch_size():
    dm = GraphDataModule.from_random_split(
        _dset(20), train_val_test_split=(0.8, 0.1, 0.1), batch_size=3
    )
    _setup(dm)
    assert dm.train_dataloader().batch_size == 3


# ---- PreSplit ---------------------------------------------------------------


def test_from_datasets_uses_provided_splits():
    train, val, test = _trivial_graphs(10), _trivial_graphs(5), _trivial_graphs(2)
    dm = GraphDataModule.from_datasets(
        PassthroughFetcher(PreSplitDataset(train=train, val=val, test=test)),
        batch_size=2,
    )
    _setup(dm)
    assert dm.data_train is train
    assert dm.data_val is val
    assert dm.data_test is test


def test_from_datasets_all_none_rejected_at_setup():
    dm = GraphDataModule.from_datasets(PassthroughFetcher(PreSplitDataset()))
    with pytest.raises(reax.exceptions.MisconfigurationException, match="all None datasets"):
        _setup(dm)


def test_from_datasets_partial_splits_ok():
    # Any subset of splits is allowed -- only the all-None case is rejected.
    train = _trivial_graphs(10)
    dm = GraphDataModule.from_datasets(
        PassthroughFetcher(PreSplitDataset(train=train, val=None, test=None))
    )
    dm.prepare_data()
    assert dm._dataset["train"] is train
    assert dm._dataset["val"] is None
    assert dm._dataset["test"] is None


@pytest.mark.parametrize("loader_attr", ["val_dataloader", "test_dataloader"])
def test_val_and_test_loaders_do_not_shuffle(loader_attr):
    dm = GraphDataModule.from_random_split(
        _dset(20), train_val_test_split=(0.8, 0.1, 0.1), batch_size=2
    )
    _setup(dm)
    dl = getattr(dm, loader_attr)()
    assert dl.shuffle is False


def test_train_dataloader_batch_shape_implicit(cube_graph: jraph.GraphsTuple):
    dataset_size = 100
    batch_size = 7
    dset = PassthroughFetcher([cube_graph for _ in range(dataset_size)])
    split = (0.8, 0.1, 0.1)

    dm = GraphDataModule.from_random_split(
        dset,
        train_val_test_split=split,
        batch_size=batch_size,
        batch_mode=gcnn.data.BatchMode.IMPLICIT,
    )
    _setup(dm)
    dl = dm.train_dataloader()

    n_train = int(dataset_size * split[0])
    expected_batches = math.ceil(n_train / batch_size)
    assert len(dl) == expected_batches

    for batch in dl:
        # Implicit: last graph is the padding placeholder
        assert batch[0].n_node.shape == (batch_size + 1,)


def test_train_dataloader_batch_shape_explicit(cube_graph: jraph.GraphsTuple):
    dataset_size = 100
    batch_size = 7
    dset = PassthroughFetcher([cube_graph for _ in range(dataset_size)])
    split = (0.8, 0.1, 0.1)

    dm = GraphDataModule.from_random_split(
        dset,
        train_val_test_split=split,
        batch_size=batch_size,
        batch_mode=gcnn.data.BatchMode.EXPLICIT,
    )
    _setup(dm)
    dl = dm.train_dataloader()

    n_train = int(dataset_size * split[0])
    expected_batches = math.ceil(n_train / batch_size)
    assert len(dl) == expected_batches

    for batch in dl:
        # Explicit: each unbatched graph has n_node of shape (1,) and
        # n_edge of shape (1,), so stacked they yield shape (batch_size, 1).
        # In this codebase graphs are stored with a leading batch axis,
        # giving n_node shape (batch_size, 2).
        assert batch[0].n_node.shape == (batch_size, 2)


# ---- k-fold -----------------------------------------------------------------


def _graph_id(g: jraph.GraphsTuple) -> int:
    """Identify a graph by the value of ``globals['g'][0]`` (set in the fixture)."""
    return int(g.globals["g"].reshape(-1)[0])


def test_kfold_graph_datamodule_setup(monkeypatch):
    dataset = _trivial_graphs(20)

    # Mock random_split to just return slices without randomness for test
    def mock_random_split(rngs, dataset, lengths):
        n_rest = 16
        return dataset[:n_rest], dataset[n_rest:]

    monkeypatch.setattr("reax.data.random_split", mock_random_split)

    gdm = GraphDataModule.from_kfold(
        PassthroughFetcher(dataset),
        test_fraction=0.2,
        batch_size=2,
        fold=0,
        n_folds=4,
        seed=42,
    )

    _setup(gdm)

    assert len(gdm.data_test) == 4
    assert len(gdm.data_val) == 4
    assert len(gdm.data_train) == 12

    # Test split must equal the 4 graphs we asked ``random_split`` to return
    assert sorted(map(_graph_id, gdm.data_test)) == [16, 17, 18, 19]


def test_kfold_graph_datamodule_coverage(monkeypatch):
    dataset = _trivial_graphs(10)

    def mock_random_split(rngs, dataset, lengths):
        return dataset[:8], dataset[8:]

    monkeypatch.setattr("reax.data.random_split", mock_random_split)

    val_ids = []

    # 4 folds over 8 samples -> 2 per fold
    for fold in range(4):
        gdm = GraphDataModule.from_kfold(
            PassthroughFetcher(dataset),
            test_fraction=0.2,
            batch_size=2,
            fold=fold,
            n_folds=4,
            seed=42,  # fixed seed
        )
        _setup(gdm)

        val_ids.extend(map(_graph_id, gdm.data_val))

    # Every training-candidate graph must have been a validation
    # target exactly once across all folds
    assert len(val_ids) == 8
    assert sorted(val_ids) == list(range(8))


def test_kfold_fold_out_of_range_rejected():
    with pytest.raises(ValueError, match="fold must be in"):
        GraphDataModule.from_kfold(
            PassthroughFetcher(_trivial_graphs(20)),
            test_fraction=0.2,
            batch_size=2,
            fold=4,  # out of range [0, 4)
            n_folds=4,
            seed=42,
        )


def test_kfold_split_validates_fold_out_of_range():
    # The validation lives on the strategy itself, not buried in setup().
    with pytest.raises(ValueError, match="fold must be in"):
        KFoldSplit(fold=4, n_folds=4)
