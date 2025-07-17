import logging
from typing import Final

import hydra
import omegaconf
import reax

from . import keys
from .. import training

_LOGGER = logging.getLogger(__name__)


DEFAULT_CONFIG_FILE: Final[str] = "config.yaml"
DEFAULT_CKPT_FILE: Final[str] = "params.ckpt"


def load_module(
    config_path: str = DEFAULT_CONFIG_FILE,
    ckpt_path: str = DEFAULT_CKPT_FILE,
    checkpointing: reax.training.Checkpointing = None,
    return_config: bool = False,
) -> training.ReaxModule | tuple[training.ReaxModule, omegaconf.DictConfig]:
    """
    Load a REAX module from a YAML configuration file, optionally restoring parameters
    from a checkpoint.

    This function uses Hydra to instantiate a module from a config file and optionally
    loads learned parameters from a checkpoint file using the provided or default
    checkpointing mechanism. It can also return the full configuration object if needed.

    Args:
        config_path (str): Path to the YAML configuration file specifying the model.
        ckpt_path (str): Path to the checkpoint file containing saved parameters.
        checkpointing (Checkpointing, optional): A Checkpointing instance to use for
            loading parameters. If None, the default REAX checkpointing is used.
        return_config (bool): If True, also return the loaded configuration object.

    Returns:
        Module or (Module, DictConfig): The instantiated REAX module, optionally
        accompanied by the loaded configuration.
    """
    cfg = omegaconf.OmegaConf.load(config_path)

    _LOGGER.info(
        "Instantiating model <%s>",
        cfg[keys.MODEL]._target_,  # pylint: disable=protected-access
    )
    module: training.ReaxModule = hydra.utils.instantiate(cfg[keys.MODEL], _convert_="object")

    if ckpt_path:
        if checkpointing is None:
            checkpointing = reax.training.get_default_checkpointing()

        ckpt = checkpointing.load(ckpt_path)
        module.set_parameters(ckpt["params"])

    if return_config:
        return module, cfg

    return module
