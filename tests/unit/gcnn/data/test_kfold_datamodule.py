import jraph
import pytest

from tensorial.gcnn.data import GraphDataModule


def test_kfold_graph_datamodule_setup(monkeypatch):
    # Create dummy dataset of 20 elements
    dataset = list(range(20))

    # Mock random_split to just return slices without randomness for test
    def mock_random_split(rngs, dataset, lengths):
        n_rest = 16
        return dataset[:n_rest], dataset[n_rest:]

    monkeypatch.setattr("reax.data.random_split", mock_random_split)

    # 20 items, (0.8, 0.0, 0.2) test split -> 16 rest, 4 test.
    # 16 items / 4 folds -> 4 items per val set.
    gdm = GraphDataModule(
        dataset=dataset,
        train_val_test_split=(0.8, 0.0, 0.2),
        batch_size=2,
        kfold=0,
        n_folds=4,
        seed=42,
    )

    class MockStage:
        engine = None

    # We patch max_padding since it does batching logic we don't care about here
    monkeypatch.setattr(
        "tensorial.gcnn.data._batching.GraphBatcher.calculate_padding", lambda *args: None
    )
    monkeypatch.setattr("tensorial.gcnn.data._batching.max_padding", lambda *args: None)

    gdm.setup(MockStage())

    assert len(gdm.data_test) == 4
    assert len(gdm.data_val) == 4
    assert len(gdm.data_train) == 12


def test_kfold_graph_datamodule_coverage(monkeypatch):
    dataset = list(range(10))

    def mock_random_split(rngs, dataset, lengths):
        return dataset[:8], dataset[8:]

    monkeypatch.setattr("reax.data.random_split", mock_random_split)
    monkeypatch.setattr(
        "tensorial.gcnn.data._batching.GraphBatcher.calculate_padding", lambda *args: None
    )
    monkeypatch.setattr("tensorial.gcnn.data._batching.max_padding", lambda *args: None)

    class MockStage:
        engine = None

    val_sets = []

    # 4 folds over 8 samples -> 2 per fold
    for fold in range(4):
        gdm = GraphDataModule(
            dataset=dataset,
            train_val_test_split=(0.8, 0.0, 0.2),
            batch_size=2,
            kfold=fold,
            n_folds=4,
            seed=42,  # fixed seed
        )
        gdm.setup(MockStage())

        val_sets.extend(list(gdm.data_val))

    # We should have seen every element exactly once in validation
    assert len(val_sets) == 8
    assert set(val_sets) == set(range(8))
