import logging
from typing import Any, Union
import warnings

_LOGGER = logging.getLogger(__name__)


def _warn(message: Union[str, Warning], stacklevel: int = 2, **kwargs: Any) -> None:
    warnings.warn(message, stacklevel=stacklevel, **kwargs)


def warn(message: Union[str, Warning], stacklevel: int = 4, **kwargs: Any) -> None:
    """Emit a warning."""
    _warn(message, stacklevel=stacklevel, **kwargs)
