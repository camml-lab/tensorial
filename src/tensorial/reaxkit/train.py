"""Training pipeline for reactive-forcefield / GNN models in tensorial.

Instantiates everything (listeners, loggers, trainer, datamodule, model) from a
Hydra/OmegaConf config, then runs the optional ``from_data`` stage and the
fit / test / predict stages.
"""

import pathlib

import hydra
from hydra.core import hydra_config
import omegaconf
from reax.lightning import rank_zero
import reax.utils

from tensorial import config as tconfig

from . import config, from_data, keys, utils

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

    rank_zero.rank_zero_debug("Instantiating listeners...")
    listeners: list[reax.TrainerListener] = utils.instantiate_listeners(cfg.get("listeners"))

    rank_zero.rank_zero_debug("Instantiating loggers...")
    logger: list[reax.Logger] = utils.instantiate_loggers(cfg.get("logger"))

    rank_zero.rank_zero_debug(
        "Instantiating trainer <%s>",
        cfg[keys.TRAINER]._target_,
    )
    trainer: reax.Trainer = tconfig.instantiate(
        cfg[keys.TRAINER],
        _convert_="none",
        listeners=listeners,
        logger=logger,
        default_root_dir=output_dir,
        # The single seed from which reax derives all randomness (JAX, numpy and python.random).
        # `None` draws a random seed, available afterwards as `trainer.seed`.
        seed=cfg.get("seed"),
    )

    rank_zero.rank_zero_debug(
        "Instantiating datamodule <%s>",
        cfg.data._target_,
    )
    datamodule: reax.DataModule = tconfig.instantiate(cfg.data)

    if cfg.get(keys.FROM_DATA):
        from_data_stage = from_data.FromData(
            cfg[keys.FROM_DATA], trainer.engine, datamodule=datamodule
        )
        stage = trainer.run(from_data_stage)
        _print_from_data(stage.calculated)

    # Save the configuration file here, this way things like inputs used to setup the model
    # will be baked into the input
    if output_dir is not None:
        _save_config(cfg, output_dir)

    rank_zero.rank_zero_debug(
        "Instantiating model <%s>",
        cfg[keys.MODEL]._target_,
    )
    model: reax.Module = tconfig.instantiate(cfg[keys.MODEL])

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "listeners": listeners,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        rank_zero.rank_zero_debug("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    # Fit the potential
    if cfg.get("train"):
        rank_zero.rank_zero_info("Starting training!")
        trainer.fit(
            model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"), **cfg.get("train")
        )

    train_metrics = trainer.listener_metrics

    if cfg.get("test"):
        rank_zero.rank_zero_info("Starting testing!")
        ckpt_path = trainer.checkpoint_listener.best_model_path
        if ckpt_path == "":
            rank_zero.rank_zero_warn("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(
            model,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
        )
        rank_zero.rank_zero_info("Best ckpt path: %s", ckpt_path)

    test_metrics = trainer.listener_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@rank_zero.rank_zero_only
def _print_from_data(calculated) -> None:
    print(
        "Calculated from data (these can be used in your config files using ${from_data.<name>}:",
    )
    utils.rich_utils.print_tree(calculated, keys.FROM_DATA)


@rank_zero.rank_zero_only
def _save_config(cfg: omegaconf.DictConfig, output_dir: pathlib.Path) -> None:
    with open(output_dir / config.DEFAULT_CONFIG_FILE, "w", encoding="utf-8") as file:
        file.write(omegaconf.OmegaConf.to_yaml(cfg, resolve=True))


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
        **tconfig.hydra_main_kwargs(),
    )(main)
    runner()
