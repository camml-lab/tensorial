from typing import Any

import hydra
import omegaconf

__all__ = ("instantiate",)


def instantiate(cfg: omegaconf.OmegaConf, **kwargs) -> Any:
    """Given an omegaconf configuration, instantiate the corresponding object"""
    return hydra.utils.instantiate(cfg, _convert_="object", **kwargs)
