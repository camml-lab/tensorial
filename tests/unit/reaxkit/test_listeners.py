import logging
import pathlib

from flax import linen
import numpy as np
import optax
import reax

from tensorial import reaxkit
from tensorial.reaxkit._metrics_printer import MetricsPrinter


class DummyModel(linen.Module):
    @linen.compact
    def __call__(self, x):
        return x**2


def _module(**overrides):
    defaults = {}
    if "output" not in overrides:
        defaults["output"] = ["predictions", "targets"]
    return reaxkit.ReaxModule(
        DummyModel(),
        loss_fn=lambda x, y: optax.l2_loss(x, y).sum(),
        optimizer=optax.adamw(learning_rate=0.01),
        **overrides,
        **defaults,
    )


def _printer_messages(caplog):
    messages = [
        record.message
        for record in caplog.records
        if record.name == "tensorial.reaxkit._metrics_printer"
    ]
    headers = [message for message in messages if "update" in message]
    rows = [message for message in messages if "update" not in message]
    return headers, rows


def test_parity_plotter_fit_writes_per_epoch_plots(tmp_path):
    module = _module()
    plotter = reaxkit.ParityPlotter()
    trainer = reax.Trainer(default_root_dir=tmp_path, listeners=[plotter])

    dataset = np.random.rand(2, 10)
    trainer.fit(module, train_dataloaders=dataset, val_dataloaders=dataset, max_epochs=2)

    plots = pathlib.Path(trainer.log_dir) / "plots"
    assert (plots / "train_epoch_1.pdf").exists()
    assert (plots / "validation_epoch_1.pdf").exists()


def test_parity_plotter_standalone_validate(tmp_path):
    module = _module()
    plotter = reaxkit.ParityPlotter()
    trainer = reax.Trainer(default_root_dir=tmp_path, listeners=plotter)

    trainer.validate(module, dataloaders=np.random.rand(10))

    assert (pathlib.Path(trainer.log_dir) / "plots" / "validation.pdf").exists()
    assert not (pathlib.Path(trainer.log_dir) / "plots" / "validation_epoch_0.pdf").exists()


def test_parity_plotter_standalone_test(tmp_path):
    module = _module()
    plotter = reaxkit.ParityPlotter()
    trainer = reax.Trainer(default_root_dir=tmp_path, listeners=plotter)

    trainer.test(module, dataloaders=np.random.rand(10))

    assert (pathlib.Path(trainer.log_dir) / "plots" / "test.pdf").exists()
    assert not (pathlib.Path(trainer.log_dir) / "plots" / "test_epoch_0.pdf").exists()


def test_parity_plotter_standalone_predict(tmp_path):
    plotter = reaxkit.ParityPlotter()
    trainer = reax.Trainer(default_root_dir=tmp_path, listeners=plotter)

    trainer.predict(_module(), dataloaders=np.random.rand(10))

    assert (pathlib.Path(trainer.log_dir) / "plots" / "predict.pdf").exists()
    assert not (pathlib.Path(trainer.log_dir) / "plots" / "predict_epoch_0.pdf").exists()


def test_parity_should_collect():
    plotter = reaxkit.ParityPlotter(fit_plot_every=10)
    assert plotter._should_collect("train", None)
    assert plotter._should_collect("validation", None)
    assert plotter._should_collect("train", 0)
    assert plotter._should_collect("train", 9)
    assert not plotter._should_collect("train", 5)
    assert not plotter._should_collect("validation", 5)
    assert plotter._should_collect("test", 1)
    assert plotter._should_collect("predict", 0)
    assert plotter._should_collect("predict", None)


def test_metrics_printer_structure_with_epoch():
    printer = MetricsPrinter()
    printer._initialize_metrics_structure({"score": 1.0}, show_epoch=True)
    assert "epoch" in printer._metrics
    assert "update" in printer._metrics
    assert "score" in printer._metrics


def test_metrics_printer_structure_without_epoch():
    printer = MetricsPrinter()
    printer._initialize_metrics_structure({"score": 1.0}, show_epoch=False)
    assert "epoch" not in printer._metrics
    assert "update" in printer._metrics
    assert "score" in printer._metrics


def test_metrics_printer_run_fit_and_standalone(tmp_path, caplog):
    module = _module(metrics={"score": "mean"})
    printer = MetricsPrinter(log_level=logging.INFO)
    trainer = reax.Trainer(
        default_root_dir=tmp_path,
        enable_progress_bar=True,
        listeners=[reaxkit.ParityPlotter(), printer],
    )

    dataset = np.random.rand(2, 10)
    with caplog.at_level(logging.INFO, logger="tensorial.reaxkit._metrics_printer"):
        trainer.fit(module, train_dataloaders=dataset, max_epochs=1)
        trainer.validate(module, dataloaders=dataset)

    joined = " ".join([record.message for record in caplog.records])
    assert "score" in joined.lower()
    # A fit loop reports an epoch index, a standalone run reports one without it.
    assert "epoch" in joined.lower()


def test_metrics_printer_prints_one_row_per_epoch_with_validation(tmp_path, caplog):
    module = _module(metrics={"score": "mean"})
    printer = MetricsPrinter(log_level=logging.INFO, log_every=1)
    trainer = reax.Trainer(default_root_dir=tmp_path, enable_progress_bar=True, listeners=[printer])

    dataset = np.random.rand(2, 10)
    with caplog.at_level(logging.INFO, logger="tensorial.reaxkit._metrics_printer"):
        trainer.fit(module, train_dataloaders=dataset, val_dataloaders=dataset, max_epochs=3)

    headers, rows = _printer_messages(caplog)

    assert len(headers) == 1
    # The columns are initialised from the training loop, so training metrics are shown
    # and not just the validation ones.
    assert "train/score" in headers[0]
    assert "val/score" in headers[0]
    # Exactly one row per epoch: no duplicated rows from the nested validation loop and
    # no sanity-check row (epoch -1).
    assert [int(row.split()[0]) for row in rows] == [0, 1, 2]


def test_metrics_printer_standalone_validate_has_no_epoch_column(tmp_path, caplog):
    module = _module(metrics={"score": "mean"})
    printer = MetricsPrinter(log_level=logging.INFO)
    trainer = reax.Trainer(default_root_dir=tmp_path, enable_progress_bar=True, listeners=[printer])

    dataset = np.random.rand(2, 10)
    with caplog.at_level(logging.INFO, logger="tensorial.reaxkit._metrics_printer"):
        trainer.validate(module, dataloaders=dataset)

    headers, rows = _printer_messages(caplog)

    assert len(headers) == 1
    assert "epoch" not in headers[0].lower()
    assert "val/score" in headers[0]
    assert len(rows) == 1
