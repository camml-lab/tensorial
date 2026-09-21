"""Hydra/OmegaConf configuration helpers.

A thin wrapper over :mod:`hydra.utils` for building objects from
:mod:`omegaconf` configuration trees, so that tensorial (and reaxkit) config
files can be materialised with a single, consistent call.
"""

from typing import Any

import hydra
import omegaconf

__all__ = ("instantiate",)


def instantiate(cfg: omegaconf.OmegaConf, **kwargs) -> Any:
    """Instantiate an object described by an OmegaConf configuration.

    Thin wrapper around :func:`hydra.utils.instantiate` that always materialises
    nested configuration into plain objects (``_convert_="object"``) before
    calling into the target class.

    Args:
        cfg: the OmegaConf node (usually a ``DictConfig``) describing the
            object to build.  Typically this must carry an
            ``_target_`` key naming the class to instantiate.
        **kwargs: any extra arguments to forward to the target's
            constructor; these override any values already present in ``cfg``.

    Returns:
        the instantiated object, as produced by ``hydra.utils.instantiate``.
    """
    return hydra.utils.instantiate(cfg, _convert_="object", **kwargs)
