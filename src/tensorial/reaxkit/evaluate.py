"""Evaluation / test entry point that runs a trained model on a (test) datamodule.

Wraps the `reax` Trainer to run the test and predict stages and to write the
resulting metrics, driven by the Hydra ``eval.yaml`` config script.
"""

import logging
from typing import TYPE_CHECKING, Any

import hydra
import omegaconf
import reax

from tensorial import config as tconfig

from . import config, keys, utils

if TYPE_CHECKING:
    from tensorial import reaxkit

_LOGGER = logging.getLogger(__name__)

DEFAULT_EVAL_FILE = "eval.yaml"


@utils.task_wrapper
def evaluate(cfg: omegaconf.DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg: DictConfig configuration composed by Hydra.

    Returns:
        The metrics logged by the stages that were run, and the objects that were created.
    """
    _LOGGER.info(
        "Instantiating datamodule <%s>",
        cfg[keys.DATA]._target_,  # pylint: disable=protected-access
    )
    datamodule: reax.DataModule = tconfig.instantiate(cfg.data, _convert_="none")

    _LOGGER.info("Instantiating loggers...")
    logger: list[reax.Logger] = utils.instantiate_loggers(cfg.get(keys.LOGGER))

    _LOGGER.info(
        "Instantiating trainer <%s>",
        cfg[keys.TRAINER]._target_,  # pylint: disable=protected-access
    )
    trainer: reax.Trainer = tconfig.instantiate(cfg[keys.TRAINER], _convert_="none", logger=logger)
    model: reaxkit.ReaxModule = config.load_module(cfg[keys.CONFIG_PATH], cfg[keys.CKPT_PATH])

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        _LOGGER.info("Logging hyperparameters")
        utils.log_hyperparameters(object_dict)

    if cfg.get(keys.VALIDATION):
        _LOGGER.info("Starting validation")
        trainer.validate(
            model,
            datamodule=datamodule,
            ckpt_path=cfg[keys.CKPT_PATH],
            **_stage_kwargs(cfg, keys.VALIDATION),
        )

    if cfg.get(keys.TEST):
        _LOGGER.info("Starting testing")
        trainer.test(
            model,
            datamodule=datamodule,
            ckpt_path=cfg[keys.CKPT_PATH],
            **_stage_kwargs(cfg, keys.TEST),
        )

    if cfg.get(keys.PREDICT):
        _LOGGER.info("Starting prediction")
        trainer.predict(
            model,
            datamodule=datamodule,
            ckpt_path=cfg[keys.CKPT_PATH],
            **_stage_kwargs(cfg, keys.PREDICT),
        )

    return trainer.listener_metrics, object_dict


def _stage_kwargs(cfg: omegaconf.DictConfig, stage: str) -> dict:
    """Keyword arguments for a stage, which may be enabled with a mapping or just ``true``.

    ``true`` is what a training config uses to switch on testing after training, so a saved
    training config can be evaluated directly.
    """
    value = cfg.get(stage)
    if isinstance(value, (dict, omegaconf.DictConfig)):
        return dict(value)
    return {}


def main(cfg: omegaconf.DictConfig) -> None:
    """Main entry point for evaluation.

    Args:
        cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)
    evaluate(cfg)


if __name__ == "__main__":
    runner = hydra.main(
        version_base="1.3",
        config_path="../../configs",
        config_name=DEFAULT_EVAL_FILE,
        **tconfig.hydra_main_kwargs(),
    )(main)

    runner()
