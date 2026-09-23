import pathlib

import hydra
from hydra.core import hydra_config

import tensorial.reaxkit as rkit

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


def test_train_simple(train_simple_mlip_config: pathlib.Path, tmpdir):
    path = REPO_ROOT / "data" / "sitraj.xyz"

    with hydra.initialize(
        config_path=train_simple_mlip_config.parent.name,
        job_name="test_app",
    ):
        cfg = hydra.compose(config_name=train_simple_mlip_config.name, return_hydra_config=True)

        # Manually set the runtime fields if needed
        cfg.hydra.job.num = 0
        cfg.hydra.runtime.output_dir = str(tmpdir)
        cfg.hydra.runtime.cwd = str(pathlib.Path.cwd())

        cfg.data.dataset.path = str(path)

        hydra_config.HydraConfig().set_config(cfg)

        rkit.train.train(cfg)

        assert (tmpdir / "config.yaml").exists()
