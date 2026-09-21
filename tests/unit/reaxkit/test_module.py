from typing import Any

from flax import linen
import numpy as np
import optax
import pytest
import reax

from tensorial import reaxkit


class DummyModel(linen.Module):
    @linen.compact
    def __call__(self, x):
        return x**2


@pytest.mark.parametrize("output", [None, "predictions", "targets", ["predictions", "targets"]])
def test_module_outputs(output, tmp_path):
    class Listener(reax.TrainerListener):
        val_outputs = []

        def on_validation_batch_end(
            self,
            trainer: "reax.Trainer",
            stage: "reax.stages.Train",
            outputs: Any,
            batch: Any,
            batch_idx: int,
            /,
        ) -> None:
            if not trainer.sanity_checking:
                self.val_outputs.append(outputs)

    module = reaxkit.ReaxModule(
        DummyModel(),
        loss_fn=lambda x, y: optax.l2_loss(x, y).sum(),
        optimizer=optax.adamw(learning_rate=0.01),
        output=output,
    )

    listener = Listener()
    trainer = reax.Trainer(listeners=listener, default_root_dir=tmp_path)

    dataset = np.random.rand(2, 10)
    trainer.fit(module, train_dataloaders=dataset, val_dataloaders=dataset, max_epochs=1)

    if output is None:
        assert listener.val_outputs == [None, None]
    elif isinstance(output, str):
        assert output in listener.val_outputs[0]
    else:
        for entry in output:
            assert entry in listener.val_outputs[0]
