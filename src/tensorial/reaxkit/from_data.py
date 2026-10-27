"""A `reax` trainer stage that computes dataset statistics and bakes them into the config.

Useful to derive per-dataset quantities (e.g. normalisation terms) once, store
them under ``${from_data.<name>}``, and then reuse those values when instantiating
the model.
"""

from collections.abc import Mapping
import functools
from typing import Any, Final

import beartype
import hydra
import jaxtyping as jt
import omegaconf
import reax
import reax.utils
from typing_extensions import override

DONE_KEY = "done"

__all__ = ("FromData",)


class FromData(reax.stages.Stage):
    """A trainer stage that will populate an OmegaConf dictionary with data statistics calculated
    from metrics.
    """

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        cfg: omegaconf.DictConfig,
        engine: reax.Engine,
        *,
        dataloader: reax.DataLoader | None = None,
        datamodule: reax.DataModule | None = None,
        dataloader_name: str = "train",
        ignore_missing: bool = True,
    ):
        """Populate a hydra configurations dictionary using calculated stats

        Args:
            cfg: the configuration dictionary
            engine: the trainer engine
            dataloader: the dataloader to use
            datamodule: if no dataloader is specified, a data module can
                be used instead
            dataloader_name: the datamodule dataloader name
            ignore_missing: if `True`, any data that is needed to
                calculate a metric but is missing will be ignored, and
                that metric will not be calculated
        """
        loaders: dict[str, Any] = {dataloader_name: dataloader}
        super().__init__(
            "from_data",
            engine,
            datamanager=reax.data.create_manager(datamodule=datamodule, engine=engine, **loaders),
        )
        # Params
        self._dataset_name: Final[str] = dataloader_name
        self._ignore_missing: Final[bool] = ignore_missing

        # State
        self._cfg = cfg
        self._calculated = {}
        self._to_calculate: dict[str, Any] = self._update_stats(self._cfg)

    @property
    def dataloader(self) -> reax.DataLoader:
        """Return the :class:`reax.DataLoader` this stage loads statistics from."""
        assert self._datamanager is not None, "FromData must have a data manager"
        return self._datamanager.get_dataloader(self._dataset_name)

    @property
    def dataloaders(self) -> reax.DataLoader:
        """Dataloader function."""
        return self.dataloader

    @property
    def calculated(self) -> dict[str, Any]:
        """The dictionary holding the calculated statistics"""
        return self._calculated

    @override
    def _run(self) -> None:
        # Iterate: while there are still metrics to calculate, evaluate them on the
        # dataset, bake the results into the config, and re-derive the remaining
        # uncalculated metrics from the updated config (so that interpolations can now
        # be resolved, e.g. one metric depending on the value of another).
        while self._to_calculate:
            loop = reax.stages.StatsEpoch(
                self,
                stats=self._to_calculate,
                dataloader_name=self._dataset_name,
                ignore_missing=self._ignore_missing,
            )
            self.run_loop(loop)

            calculated: dict = self.logged_metrics
            # Convert to types that can be used by omegaconf and update the configuration with
            # the values
            calculated = {
                label: reax.utils.arrays.to_base(stat) for label, stat in calculated.items()
            }

            if self._ignore_missing:
                # Set any that we couldn't calculate to `None`
                for missing in self._to_calculate.keys() - calculated.keys():
                    calculated[missing] = None

            self._cfg.update(calculated)
            self._calculated.update(calculated)

            # Re-derive the next set to calculate from the updated config.
            self._to_calculate = self._update_stats(self._cfg)

    def _update_stats(self, from_data: Mapping) -> dict:
        with_dependencies = []

        # Find those that we will come back to for a second path
        for entry in find_iterpol(from_data):
            with_dependencies.append(entry[0][0])
        with_dependencies = set(with_dependencies)

        to_calculate = {}
        for label, value in from_data.items():
            if label in self._calculated:
                continue

            if label in with_dependencies:
                continue

            if omegaconf.OmegaConf.is_dict(value):
                stat = hydra.utils.instantiate(value, _convert_="object")
            else:
                stat = reax.metrics.get(value)

            to_calculate[label] = stat

        return to_calculate


@functools.singledispatch
def _to_omega(value):
    return value


@_to_omega.register
def _(value: dict) -> dict:
    return {key: _to_omega(value) for key, value in value.items()}


@_to_omega.register
def _(value: jt.Array) -> int | float | list:
    return reax.utils.arrays.to_base(value)


def find_iterpol(root, path=()):
    """Yield ``(path, key)`` for every nested OmegaConf interpolation in a config node.

    Args:
        root: An OmegaConf dict (or a nested dict).
        path: The partial path accumulated so far (used only on recursion).

    Yields:
        Tuples of the path to, and the key within, each interpolation found.
    """
    for key, value in root.items():
        if isinstance(value, str):
            if omegaconf.OmegaConf.is_interpolation(root, key):
                if not isinstance(root[key], (int, float, list)):
                    yield path, key
        elif omegaconf.OmegaConf.is_dict(value):
            yield from find_iterpol(value, (key,))
