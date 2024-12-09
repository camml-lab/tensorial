import functools
from typing import Callable, Optional, cast

from flax import linen
import hydra
import jax
import jaxtyping as jt
import jraph
import omegaconf
import orbax.checkpoint as ocp
import reax
from reax.modules import BatchT, OutputT_co
import reax.utils

from tensorial import config as config_

__all__ = ("TrainingModule",)

MetricsDict = dict[str, reax.Metric]
LossFn = Callable[[jraph.GraphsTuple, jraph.GraphsTuple], jax.Array]


class TrainingModule(reax.Module):
    _loss_fn: LossFn
    _metrics: Optional[reax.metrics.MetricCollection] = None
    _model: Optional[linen.Module] = None

    def __init__(self, config: omegaconf.DictConfig):
        super().__init__()
        self._cfg = config

    def create_and_init_model(self, example_inputs):
        """Create the model and initialise parameters"""
        self._model = config_.create_module(self._cfg.model)
        params = self._model.init(self.rng_key(), example_inputs)
        self.set_parameters(params)

    def load_checkpoint(self, checkpoint_path, example_inputs):
        if self._model is None:
            self.create_and_init_model(example_inputs)

        assert self.parameters() is not None
        abstract_state = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, self.parameters())
        ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
        restored = ckptr.restore(
            checkpoint_path / "1", args=ocp.args.StandardRestore(abstract_state)
        )

    def setup(self, stage: reax.Stage):
        if isinstance(stage, reax.stages.Train) and self.parameters() is None:
            # Calculate any statistics that the model will need in order to be configured
            if self._cfg.get("from_data"):
                calculate_stats(self._cfg.from_data, stage.dataloader)

            train_cfg = self._cfg.training

            # Create the loss function to be used to train
            self._loss_fn = hydra.utils.instantiate(train_cfg.loss_fn, _convert_="object")

            # Create the metrics to be used during fitting
            if train_cfg.get("metrics"):
                self._metrics = reax.metrics.MetricCollection(
                    config_.create_metrics(train_cfg.metrics)
                )

            batch = next(iter(stage.dataloader))
            inputs = batch
            if isinstance(batch, tuple):
                inputs = batch[0]

            self.create_and_init_model(inputs)
        elif self._model is None:
            # Create the model
            self._model = config_.create_module(self._cfg.model)

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
        have_metrics = metrics is not None
        self.log(
            "train.loss", loss, on_step=False, on_epoch=True, logger=True, prog_bar=not have_metrics
        )

        if metrics:
            metrics = cast(dict[str, reax.Metric], metrics)
            for name, metric in metrics.items():
                self.log(
                    f"train.{name}",
                    metric,
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    prog_bar=True,
                )

        return loss, grads

    def validation_step(self, batch: tuple[jraph.GraphsTuple, jraph.GraphsTuple], batch_idx: int):
        inputs, outputs = batch
        if outputs is None:
            outputs = inputs

        loss, metrics = self.step(
            self.parameters(), inputs, outputs, self._model.apply, self._loss_fn, self._metrics
        )
        have_metrics = metrics is not None
        self.log(
            "val.loss", loss, on_step=False, on_epoch=True, logger=True, prog_bar=not have_metrics
        )

        if have_metrics:
            metrics = cast(reax.metrics.MetricCollection, metrics)
            for name, metric in metrics.items():
                self.log(
                    f"val.{name}",
                    metric,
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    prog_bar=True,
                )

    def predict_step(self, batch: jraph.GraphsTuple, batch_idx: int) -> jraph.GraphsTuple:
        inputs, _outputs = batch
        return self._model.apply(self.parameters(), inputs)

    @staticmethod
    @functools.partial(jax.jit, static_argnums=[3, 4, 5])
    def step(
        params: jt.PyTree,
        inputs: jraph.GraphsTuple,
        _targets: jraph.GraphsTuple,
        model: Callable[[jt.PyTree, jraph.GraphsTuple], jraph.GraphsTuple],
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
    calculated = reax.evaluate_stats(to_calculate, training_data)

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

    if to_calculate:
        # Calculate the dependent statistics
        calculated = reax.evaluate_stats(to_calculate, training_data)

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
