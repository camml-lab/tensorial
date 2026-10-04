"""Logging helpers shared by the ``tensorial.reaxkit`` training pipeline."""

import logging
from typing import Any

import jax
from lightning_utilities.core import rank_zero
from omegaconf import OmegaConf

__all__ = ("log_model_size", "log_hyperparameters")

_LOGGER = logging.getLogger(__name__)


@rank_zero.rank_zero_only
def log_model_size(model: Any) -> None:
    """Log a model's parameter count and memory size at INFO level.

    Computes the total parameter count and the total size on device
    (``sum(arr.size * arr.dtype.itemsize)`` over ``jax.tree.leaves``) and
    logs a single line such as
    ``"Model size: 1,234,567 parameters (4.94 MB)"``.

    Only runs on the zeroth rank.

    Args:
        model: A JAX-compatible object exposing a ``parameters()`` method.
    """
    params = jax.tree.leaves(model.parameters())
    num_parameters = sum(arr.size for arr in params)
    size_in_bytes = sum(arr.size * arr.dtype.itemsize for arr in params)
    _LOGGER.info(
        "Model size: %s parameters (%.2f MB)",
        f"{num_parameters:,}",
        size_in_bytes / 1_000_000,
    )


@rank_zero.rank_zero_only
def log_hyperparameters(object_dict: dict[str, Any]) -> None:
    """Log the hyperparameters of a run to all active Lightning loggers.

    Only a subset of the config is forwarded and the total number of model
    parameters is computed and recorded as ``model/params/total``.

    Args:
        object_dict: A mapping with the following keys:
            - ``"cfg"``: An ``omegaconf.DictConfig`` containing the full config.
            - ``"model"``: The JAX/Lightning model to inspect.
            - ``"trainer"``: The Lightning trainer owning the loggers.
    """
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"])
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        _LOGGER.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(x.size for x in jax.tree.leaves(model.parameters()))

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)
