import pytest
import reax.testing

from tensorial import reaxkit
from tensorial.reaxkit import keys


def train(devices: int):
    data = {
        "_target_": "reax.demos.boring_classes.RandomDataset",
        "size": 32,
        "length": 32,
    }
    trainer = {
        "_target_": "reax.training.Trainer",
        "accelerator": "cpu",
        "devices": devices,
    }
    model = {"_target_": "reax.demos.boring_classes.BoringModel"}
    cfg = {keys.DATA: data, keys.MODEL: model, keys.TRAINER: trainer}

    reaxkit.train.train(cfg)


@pytest.mark.parametrize("devices", [1, 4])
def test_train(devices):
    run = train
    if devices > 1:
        run = reax.testing.in_subprocess(train)

    run(devices)
