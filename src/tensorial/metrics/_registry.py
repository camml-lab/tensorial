from tensorial import _utils

from .metric import Metric

__all__ = ("Registry", "get_registry", "set_registry", "get")

Registry = _utils.Registry[type[Metric]]


# Helpers to make it easy to choose a metric using a string
_registry = _utils.Registry[type(Metric)]()


def get_registry() -> Registry:
    global _registry  # pylint: disable=global-variable-not-assigned
    return _registry


def set_registry(registry: Registry) -> None:
    global _registry  # pylint: disable=global-statement
    _registry = registry


def get(name: str) -> type[Metric]:
    """Convenience method to get a metric using its name"""
    return get_registry()[name]
