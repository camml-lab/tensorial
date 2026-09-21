"""Training pipeline for reactive-forcefield / GNN models in tensorial.

Instantiates everything (listeners, loggers, trainer, datamodule, model) from a
Hydra/OmegaConf config, then runs the optional ``from_data`` stage and the
fit / test / predict stages.
"""

import logging
import pathlib

import hydra
from hydra.core import hydra_config
import omegaconf
import reax.utils

from . import config, from_data, keys, utils

_LOGGER = logging.getLogger(__name__)

DEFAULT_TRAIN_FILE = "train.yaml"


def train(cfg: omegaconf.DictConfig | dict):
    """Set up and run the full training / evaluation pipeline from a config.

    Args:
        cfg: A Hydra ``DictConfig`` (or plain dict) describing the experiment.
            Recognised keys include ``data``, the model, the trainer, ``train``,
            ``test``, ``predict`` and ``from_data``.

    Returns:
        A dict of the final metrics produced by the trainer.
    """
    if isinstance(cfg, dict):
        cfg = omegaconf.DictConfig(cfg)

    try:
        output_dir = pathlib.Path(hydra_config.HydraConfig.get().runtime.output_dir)
    except ValueError:
        output_dir = None

    # set seed for random number generators in JAX, numpy and python.random
    if cfg.get("seed"):
        reax.seed_everything(cfg.seed, workers=True)

    _LOGGER.debug("Instantiating listeners...")
    listeners: list[reax.TrainerListener] = utils.instantiate_listeners(cfg.get("listeners"))

    _LOGGER.debug("Instantiating loggers...")
    logger: list[reax.Logger] = utils.instantiate_loggers(cfg.get("logger"))

    _LOGGER.debug(
        "Instantiating trainer <%s>", cfg[keys.TRAINER]._target_  # pylint: disable=protected-access
    )
    trainer: reax.Trainer = hydra.utils.instantiate(
        cfg[keys.TRAINER],
        listeners=listeners,
        logger=logger,
        default_root_dir=output_dir,
    )

    _LOGGER.debug(
        "Instantiating datamodule <%s>", cfg.data._target_  # pylint: disable=protected-access
    )
    datamodule: reax.DataModule = hydra.utils.instantiate(cfg.data, _convert_="object")

    if cfg.get(keys.FROM_DATA):
        from_data_stage = from_data.FromData(  # pylint: disable=no-member
            cfg[keys.FROM_DATA], trainer.engine, rngs=trainer.rngs, datamodule=datamodule
        )
        stage = trainer.run(from_data_stage)
        print(
            "Calculated from data (these can be used in your config files using "
            "${from_data.<name>}:",
        )
        utils.rich_utils.print_tree(stage.calculated, keys.FROM_DATA)

    # Save the configuration file here, this way things like inputs used to setup the model
    # will be baked into the input
    if output_dir is not None:
        with open(output_dir / config.DEFAULT_CONFIG_FILE, "w", encoding="utf-8") as file:
            file.write(omegaconf.OmegaConf.to_yaml(cfg, resolve=True))

    _LOGGER.debug(
        "Instantiating model <%s>",
        cfg[keys.MODEL]._target_,  # pylint: disable=protected-access
    )
    model: reax.Module = hydra.utils.instantiate(cfg[keys.MODEL], _convert_="object")

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "listeners": listeners,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        _LOGGER.debug("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    # Fit the potential
    if cfg.get("train"):
        _LOGGER.info("Starting training!")
        trainer.fit(
            model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"), **cfg.get("train")
        )

    train_metrics = trainer.listener_metrics

    if cfg.get("test"):
        _LOGGER.info("Starting testing!")
        ckpt_path = trainer.checkpoint_listener.best_model_path
        if ckpt_path == "":
            _LOGGER.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(
            model,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
        )
        _LOGGER.info("Best ckpt path: %s", ckpt_path)

    test_metrics = trainer.listener_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


def main(cfg: omegaconf.DictConfig) -> float | None:
    """Main entry point for training.

    Args:
        cfg: DictConfig configuration composed by Hydra.

    Returns:
        Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    runner = hydra.main(
        version_base="1.3",
        config_path="../../configs",
        config_name=DEFAULT_TRAIN_FILE,
    )(main)
    runner()
