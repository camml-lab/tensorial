import pytest
import reax


@pytest.fixture
def test_trainer(tmp_path):
    return reax.Trainer(default_root_dir=tmp_path)
