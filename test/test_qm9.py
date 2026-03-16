import pathlib

import pytest

from tensorial.datasets.qm9 import Qm9


@pytest.fixture(scope="session")
def qm9_data_dir(pytestconfig) -> pathlib.Path:
    """Returns the path to the directory containing the test database."""
    return pytestconfig.rootpath / "test" / "assets"


def get_qm9_filenames(data_dir, limit=None, shuffle=False, rng_seed=None):
    dataset = Qm9(
        data_dir=str(data_dir), download=False, limit=limit, shuffle=shuffle, rng_seed=rng_seed
    )
    return [entry["filename"] for entry in dataset]


@pytest.mark.parametrize(
    "shuffle,rng_seed,expected_reproducible,expected_diff_from_base",
    [
        (False, None, True, False),  # Case 1: No shuffle
        (False, 42, True, False),  # Case 2: No shuffle with seed (seed ignored, order unchanged)
        (True, 42, True, True),  # Case 3: Shuffle with seed (reproducible, different from base)
        (
            True,
            None,
            False,
            True,
        ),  # Case 4: Shuffle without seed (unpredictable, different from base)
    ],
)
def test_shuffling_combinations(
    qm9_data_dir, shuffle, rng_seed, expected_reproducible, expected_diff_from_base
):
    """Verify the four main combinations of shuffle and rng_seed."""
    base_order = get_qm9_filenames(qm9_data_dir, shuffle=False)

    order1 = get_qm9_filenames(qm9_data_dir, shuffle=shuffle, rng_seed=rng_seed)
    order2 = get_qm9_filenames(qm9_data_dir, shuffle=shuffle, rng_seed=rng_seed)

    if expected_reproducible:
        assert order1 == order2
    else:
        assert order1 != order2

    if expected_diff_from_base:
        assert order1 != base_order
    else:
        assert order1 == base_order
