"""Hydra/OmegaConf configuration helpers.

A thin wrapper over :mod:`hydra.utils` for building objects from
:mod:`omegaconf` configuration trees, so that tensorial (and reaxkit) config
files can be materialised with a single, consistent call.

This module is also the single place that decides which ``_target_`` values
Hydra is allowed to execute.  From Hydra 1.4, :func:`hydra.utils.instantiate`
expects an *execution whitelist* supplied by trusted Python code (see
https://hydra.cc/docs/advanced/execution_whitelist/).  Tensorial trusts its own
packages (and the libraries its shipped configs use) by default; user code can
extend the list with :func:`allow`, or by setting the
``TENSORIAL_HYDRA_WHITELIST`` environment variable to a comma-separated list of
patterns (e.g. ``"my_package.*,other.Thing"``).  Setting it to ``UNSAFE``
disables Hydra's execution checks altogether.

Older Hydra versions (< 1.4) have no whitelist support, in which case the
whitelist is simply not passed on.
"""

import os
from typing import Any, Final

import hydra
import omegaconf

__all__ = (
    "DEFAULT_WHITELIST",
    "HAS_EXECUTION_WHITELIST",
    "UNSAFE_DISABLE_EXECUTION_CHECKS",
    "WHITELIST_ENV_VAR",
    "allow",
    "get_whitelist",
    "hydra_main_kwargs",
    "instantiate",
)

WHITELIST_ENV_VAR: Final[str] = "TENSORIAL_HYDRA_WHITELIST"
_UNSAFE_ENV_VALUE: Final[str] = "UNSAFE"

#: Target patterns that are always trusted.  A trailing ``.*`` matches everything under a package.
DEFAULT_WHITELIST: Final[tuple[str, ...]] = ("tensorial.*", "reax.*", "optax.*")

# Looked up dynamically so that static checkers don't flag it when Hydra < 1.4 is installed
#: Hydra's sentinel that disables execution checks, or ``None`` if Hydra < 1.4
UNSAFE_DISABLE_EXECUTION_CHECKS: Final[Any] = getattr(
    hydra.utils, "UNSAFE_DISABLE_EXECUTION_CHECKS", None
)

#: Whether the installed Hydra supports execution whitelists (Hydra >= 1.4)
HAS_EXECUTION_WHITELIST: Final[bool] = UNSAFE_DISABLE_EXECUTION_CHECKS is not None

_extra_whitelist: list[str] = []


def allow(*patterns: str) -> None:
    """Add target patterns to the execution whitelist used by :func:`instantiate`.

    Args:
        *patterns: fully qualified target names (e.g. ``"my_package.MyModel"``) or package
            prefixes with a trailing ``.*`` (e.g. ``"my_package.*"``).
    """
    for pattern in patterns:
        if not isinstance(pattern, str):
            raise TypeError(f"Whitelist patterns must be strings, got {type(pattern).__name__}")
        if pattern not in _extra_whitelist:
            _extra_whitelist.append(pattern)


def get_whitelist() -> Any:
    """Get the execution whitelist to pass to Hydra.

    Returns:
        a tuple of target patterns made up of :data:`DEFAULT_WHITELIST`, any patterns registered
        via :func:`allow` and any from the ``TENSORIAL_HYDRA_WHITELIST`` environment variable, or
        :data:`UNSAFE_DISABLE_EXECUTION_CHECKS` if that variable is set to ``UNSAFE``.
    """
    env_patterns = [
        entry.strip() for entry in os.environ.get(WHITELIST_ENV_VAR, "").split(",") if entry.strip()
    ]
    if _UNSAFE_ENV_VALUE in env_patterns:
        return UNSAFE_DISABLE_EXECUTION_CHECKS

    whitelist = list(DEFAULT_WHITELIST)
    for pattern in [*_extra_whitelist, *env_patterns]:
        if pattern not in whitelist:
            whitelist.append(pattern)
    return tuple(whitelist)


def hydra_main_kwargs() -> dict[str, Any]:
    """Get the keyword arguments to pass to :func:`hydra.main` so that it uses the whitelist.

    Returns:
        ``{"execution_whitelist": ...}`` on Hydra >= 1.4, and an empty dictionary otherwise.
    """
    if not HAS_EXECUTION_WHITELIST:
        return {}
    return {"execution_whitelist": get_whitelist()}


def instantiate(
    cfg: omegaconf.DictConfig | omegaconf.ListConfig | dict | list,
    *args,
    _convert_: str = "object",
    **kwargs,
) -> Any:
    """Instantiate an object described by an OmegaConf configuration.

    Thin wrapper around :func:`hydra.utils.instantiate` that, by default, materialises nested
    configuration into plain objects (``_convert_="object"``) before calling into the target
    class, and only allows targets in the tensorial execution whitelist (see
    :func:`get_whitelist`).

    Args:
        cfg: the OmegaConf node (usually a ``DictConfig``) describing the
            object to build.  Typically this must carry an
            ``_target_`` key naming the class to instantiate.
        *args: positional arguments to forward to the target.
        _convert_: the Hydra conversion strategy (``"none"``, ``"partial"``, ``"object"`` or
            ``"all"``).
        **kwargs: any extra arguments to forward to the target's
            constructor; these override any values already present in ``cfg``.
            An explicit ``_execution_whitelist_`` overrides the tensorial whitelist.

    Returns:
        the instantiated object, as produced by ``hydra.utils.instantiate``.
    """
    if HAS_EXECUTION_WHITELIST:
        kwargs.setdefault("_execution_whitelist_", get_whitelist())
    return hydra.utils.instantiate(cfg, *args, _convert_=_convert_, **kwargs)
