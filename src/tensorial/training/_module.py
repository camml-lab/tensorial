import functools
from typing import Any, Callable, Optional, cast

from flax import linen
import hydra
import jax
import jraph
import omegaconf
import reax
import reax.utils

from tensorial import config as config_

__all__ = ("TrainingModule",)

MetricsDict = dict[str, reax.Metric]
LossFn = Callable[[jraph.GraphsTuple, jraph.GraphsTuple], jax.Array]


class TrainingModule(reax.Module):
    _loss_fn: LossFn
    _metrics: Optional[reax.metrics.MetricCollection] = None
    _model: linen.Module

    def __init__(self, config: omegaconf.DictConfig):
        super().__init__()
        self._cfg = config

    def setup(self, stage: str):
        if stage == "training" and self.parameters() is None:
            # Calculate any statistics that the model will need to be configured
            if self._cfg.get("from_data"):
                calculate_stats(self._cfg.from_data, self.trainer.train_dataloader)

            train_cfg = self._cfg.training

            # Create the loss function to be used to train
            self._loss_fn = hydra.utils.instantiate(train_cfg.loss_fn, _convert_="object")

            # Create the metrics to be used during fitting
            if train_cfg.get("metrics"):
                self._metrics = reax.metrics.MetricCollection(create_metrics(train_cfg.metrics))

            # Create and initialise the model
            self._model = config_.create_module(self._cfg.model)

            batch = next(iter(self.trainer.train_dataloader))
            inputs = batch
            if isinstance(batch, tuple):
                inputs = batch[0]
            params = self._model.init(self.rng_key(), inputs)
            self.set_parameters(params)

    def configure_optimizers(self):
        assert self.parameters() is not None  # nosec B101
        optimiser = hydra.utils.instantiate(self._cfg.training.optimiser, _convert_="object")
        state = optimiser.init(self.parameters())
        return optimiser, state

    def training_step(
        self, batch: tuple[jraph.GraphsTuple, jraph.GraphsTuple], batch_idx: int
    ) -> tuple[jax.Array, jax.Array]:
        inputs, outputs = batch
        if outputs is None:
            outputs = inputs

        (loss, metrics), grads = jax.value_and_grad(self.step, argnums=0, has_aux=True)(
            self.parameters(), inputs, outputs, self._model.apply, self._loss_fn, self._metrics
        )
        self.log("loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        if metrics:
            metrics = cast(dict[str, reax.Metric], metrics)
            for name, metric in metrics.items():
                self.log(name, metric, on_step=True, prog_bar=True)

        return loss, grads

    def validation_step(self, batch: tuple[jraph.GraphsTuple, jraph.GraphsTuple], batch_idx: int):
        inputs, outputs = batch
        if outputs is None:
            outputs = inputs

        loss, metrics = self.step(
            self.parameters(), inputs, outputs, self._model.apply, self._loss_fn, self._metrics
        )
        self.log("loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        if metrics is not None:
            metrics = cast(reax.metrics.MetricCollection, metrics)
            for name, metric in metrics.items():
                self.log(name, metric, on_step=True, prog_bar=True)

    @staticmethod
    @functools.partial(jax.jit, static_argnums=[3, 4, 5])
    def step(
        params,
        inputs: jraph.GraphsTuple,
        _targets: jraph.GraphsTuple,
        model: Callable[[Any, jraph.GraphsTuple], jraph.GraphsTuple],
        loss_fn: Callable,
        metrics: Optional[reax.metrics.MetricCollection] = None,
    ) -> tuple[jax.Array, Optional[reax.metrics.MetricCollection]]:
        """Calculate loss and, optionally metrics"""
        predictions = model(params, inputs)
        if metrics:
            metrics = metrics.create(predictions, inputs)

        return loss_fn(predictions, inputs), metrics

    @staticmethod
    @functools.partial(jax.jit, static_argnums=2)
    def calculate_metrics(
        predictions: jraph.GraphsTuple, targets: jraph.GraphsTuple, metrics: MetricsDict
    ) -> dict[str, reax.Metric]:
        return {key: metric.create(predictions, targets) for key, metric in metrics.items()}


def calculate_stats(
    from_data: omegaconf.DictConfig, training_data: reax.DataLoader[jraph.GraphsTuple]
):
    """This does an inplace update of the from_data config"""
    with_dependencies = []

    # Find those that we will come back to for a second path
    for entry in find_iterpol(from_data):
        with_dependencies.append(entry[0][0])
    with_dependencies = set(with_dependencies)

    to_calculate = {}
    for label, value in from_data.items():
        if label in with_dependencies:
            continue

        if omegaconf.OmegaConf.is_dict(value):
            stat = hydra.utils.instantiate(value, _convert_="object")
        else:
            stat = reax.metrics.get(value)

        to_calculate[label] = stat

    # Calculate the statistics
    calculated = reax.metrics.evaluate_stats(to_calculate, training_data)

    # Convert to types that can be used by omegaconf and update the configuration with the values
    calculated = {label: reax.utils.arrays.to_base(stat) for label, stat in calculated.items()}

    from_data.update(calculated)

    to_calculate = {}
    for label in with_dependencies:
        value = from_data[label]
        if omegaconf.OmegaConf.is_dict(value):
            stat = hydra.utils.instantiate(value, _convert_="object")
        else:
            stat = reax.metrics.get(value)

        to_calculate[label] = stat

    # Calculate the statistics
    calculated = reax.metrics.evaluate_stats(to_calculate, training_data)

    # Convert to types that can be used by omegaconf and update the configuration with the values
    calculated = {label: reax.utils.arrays.to_base(stat) for label, stat in calculated.items()}

    from_data.update(calculated)


def find_iterpol(root, path=()):
    for key, value in root.items():
        if isinstance(value, str):
            if omegaconf.OmegaConf.is_interpolation(root, key):
                yield path, key
        elif omegaconf.OmegaConf.is_dict(value):
            yield from find_iterpol(value, (key,))


def create_metrics(metrics: omegaconf.DictConfig) -> MetricsDict:
    """Create all the metrics from the configuration"""
    found: dict[str, reax.Metric] = {}
    for label, metric_name in metrics.items():
        try:
            metric = reax.metrics.get(metric_name)
        except KeyError:
            raise ValueError(f"Unknown metric: {label} {metric_name}") from None

        found[label] = metric

    return found
