from typing import TYPE_CHECKING

import hydra
import omegaconf
import reax

from . import config, keys, utils

if TYPE_CHECKING:
    from tensorial import reaxkit

_LOGGER = utils.RankedLogger(__name__, rank_zero_only=True)

DEFAULT_EVAL_FILE = "eval.yaml"


@utils.task_wrapper
def evaluate(cfg: omegaconf.DictConfig) -> None:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg: DictConfig configuration composed by Hydra.
    """
    _LOGGER.info(
        "Instantiating datamodule <%s>",
        cfg[keys.DATA]._target_,  # pylint: disable=protected-access
    )
    datamodule: reax.DataModule = hydra.utils.instantiate(cfg.data)

    _LOGGER.info("Instantiating loggers...")
    logger: list[reax.Logger] = utils.instantiate_loggers(cfg.get(keys.LOGGER))

    _LOGGER.info(
        "Instantiating trainer <%s>",
        cfg[keys.TRAINER]._target_,  # pylint: disable=protected-access
    )
    trainer: reax.Trainer = hydra.utils.instantiate(cfg[keys.TRAINER], logger=logger)
    model: "reaxkit.ReaxModule" = config.load_module(cfg[keys.CONFIG_PATH], cfg[keys.CKPT_PATH])

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
            **cfg.get(keys.VALIDATION, {}),
        )

    if cfg.get(keys.TEST):
        _LOGGER.info("Starting testing")
        trainer.test(
            model, datamodule=datamodule, ckpt_path=cfg[keys.CKPT_PATH], **cfg.get(keys.TEST, {})
        )

    if cfg.get(keys.PREDICT):
        _LOGGER.info("Starting prediction")
        trainer.predict(
            model, datamodule=datamodule, ckpt_path=cfg[keys.CKPT_PATH], **cfg.get(keys.PREDICT, {})
        )


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
    )(main)

    runner()
