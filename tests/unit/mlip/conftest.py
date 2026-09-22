import pathlib

import pytest

CONFIG_PATH = pathlib.Path(__file__).parent / "configs"


@pytest.fixture
def train_simple_mlip_config() -> pathlib.Path:
    return CONFIG_PATH / "train_simple_mlip.yaml"
