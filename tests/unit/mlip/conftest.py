import pathlib
from typing import Final

import pytest

CONFIG_PATH: Final[pathlib.Path] = pathlib.Path(__file__).parent / "configs"
CONFIG_FILE: Final[str] = "train_simple_mlip.yaml"


@pytest.fixture
def train_simple_mlip_config() -> pathlib.Path:
    return CONFIG_PATH / CONFIG_FILE


@pytest.fixture(scope="session")
def mlip_si_data(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Materialise the ``mlip.si`` example and return its dataset path.

    ``sitraj.xyz`` is not committed (``/data/`` is gitignored), so the file
    is produced on demand by ``examples init``, which copies the path
    declared in ``si.yaml``'s ``# requires:`` comment into ``data/``.
    """
    from tensorial.reaxkit import cli

    init_dir = tmp_path_factory.mktemp("mlip_si", numbered=False)
    cli._init_example("mlip.si", init_dir)
    data_file = init_dir / "mlip-si" / "data" / "sitraj.xyz"
    if not data_file.is_file():
        raise FileNotFoundError(f"'examples init mlip.si' did not produce {data_file}")
    return data_file
