import copy
import logging
import numbers
import sys
from typing import Final

import jax
from lightning_utilities.core import rank_zero
import reax
import tqdm
from typing_extensions import override

from .utils import pylogger

__all__ = ("MetricsPrinter",)

_LOGGER = pylogger.RankedLogger(__name__, rank_zero_only=True)


MAX_PRE_DECIMAL_DIGITS: Final[int] = 4


class MetricsPrinter(reax.listeners.ProgressBar):
    """
    Prints all scalar metrics found in trainer.progress_bar_metrics with
    dynamic column width based on the title or the value.
    """

    _progress_bar = None
    NUMBER_DECIMALS = 5
    MIN_COLUMN_WIDTH = 5

    def __init__(
        self,
        log_level=logging.INFO,
        log_every: int = 10,
    ):
        super().__init__()

        self._log_level = log_level
        self._log_every = log_every

        # These will be set dynamically once metrics are available
        self._metrics = None
        self._widths = {}
        self._line = None

        # Stores the format string and dict for the header line
        self._header_format = None
        self._header_values = None

        self._enabled = True
        self._header_printed = False

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def is_disabled(self) -> bool:
        return not self.is_enabled

    @override
    def enable(self) -> None:
        self._enabled = True

    @override
    def disable(self) -> None:
        self._enabled = False
        self._header_printed = False

    def init_train_tqdm(self, stage: "reax.stages.EpochStage") -> tqdm.tqdm:
        """Override this to customize the tqdm bar for training."""
        return tqdm.tqdm(
            desc=stage.name,
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
        )

    @override
    def on_stage_start(self, _trainer: "reax.Trainer", stage: "reax.stages.Stage", /) -> None:
        if isinstance(stage, reax.stages.EpochStage):
            if self._progress_bar is not None:
                self._progress_bar.close()
            self._progress_bar = self.init_train_tqdm(stage)
            self._progress_bar.reset(total=stage.max_iters)
            self._progress_bar.initial = 0

    @override
    def on_stage_end(self, trainer: "reax.Trainer", stage: "reax.Stage", /) -> None:
        if self._progress_bar is not None:
            self._progress_bar.close()

        if isinstance(stage, reax.stages.FitEpoch) and self._enabled:
            # Check if metrics structure needs to be initialized (i.e., first time running)
            if not self._metrics:
                self._initialize_metrics_structure(trainer.progress_bar_metrics)

            if trainer.current_epoch % self._log_every == 0:
                # Print header using the robust logging method
                if not self._header_printed:
                    # Log the header as a format string with its values
                    self.do_log(self._header_format, self._header_values)
                    self._header_printed = True

                msg, values = self._get_metrics(trainer)
                self.do_log(msg, values)

    @override
    def on_stage_iter_start(
        self, _trainer: "reax.Trainer", stage: "reax.Stage", _step: int, /
    ) -> None:
        if isinstance(stage, reax.stages.EpochStage) and self._progress_bar is not None:
            pbar = self._progress_bar
            pbar.n = stage.iteration + 1
            pbar.refresh()

    def _get_float_format_width(self) -> int:
        """Calculates the minimum required width for the floating point value."""
        # This formula calculates the width for a string like: '-1234.56789'
        return 1 + 1 + self.NUMBER_DECIMALS + MAX_PRE_DECIMAL_DIGITS

    def _initialize_metrics_structure(self, initial_metrics: dict) -> None:
        """Dynamically sets up the list of metrics, header, line format, and column widths."""

        float_width = self._get_float_format_width()

        # Lists and dicts to build the format strings
        header_formats = []
        header_values = {}
        line_formats = []
        metric_names = []
        widths = {}

        # --- Fixed Columns: Epoch and Updates ---

        # Epoch Column: Ensure alignment with data format (e.g., %(epoch)5i)
        name_epoch = "epoch"
        width_epoch = max(len(name_epoch), 5)
        widths[name_epoch] = width_epoch
        metric_names.append(name_epoch)
        # Header format: placeholder to log the centered string
        header_formats.append(f"%({name_epoch}){width_epoch}s")
        header_values[name_epoch] = name_epoch.center(width_epoch)
        # Data format: integer
        line_formats.append(f"%({name_epoch}){width_epoch}i")

        # Update Column
        name_update = "update"  # Use 'update' for the key name consistently
        width_update = max(len(name_update), 7)
        widths[name_update] = width_update
        metric_names.append(name_update)
        # Header format: placeholder to log the centered string
        header_formats.append(f"%({name_update}){width_update}s")
        header_values[name_update] = name_update.center(width_update)
        # Data format: integer
        line_formats.append(f"%({name_update}){width_update}i")

        # --- Dynamic Scalar Metrics ---

        all_metrics_keys = sorted(
            [
                key
                for key, value in initial_metrics.items()
                if key not in ("epoch", "update") and is_scalar(value)
            ]
        )

        for name in all_metrics_keys:
            required_width = max(len(name), float_width, self.MIN_COLUMN_WIDTH)

            widths[name] = required_width
            metric_names.append(name)

            # Header format: placeholder to log the centered string
            header_formats.append(f"%({name}){required_width}s")
            header_values[name] = name

            # Data format: floating point
            line_formats.append(f"%({name}){required_width}.{self.NUMBER_DECIMALS}f")

        # 4. Store the results
        self._metrics = metric_names
        self._line = line_formats
        self._widths = widths

        # Store the header components for robust logging
        self._header_format = " ".join(header_formats)
        self._header_values = header_values

    def _get_metrics(self, trainer: "reax.Trainer") -> tuple[str, dict[str, jax.Array | str]]:

        if not self._metrics or not self._line:
            return "[Metrics not initialized]", {}

        metrics = copy.deepcopy(trainer.progress_bar_metrics)
        metrics.setdefault("epoch", trainer.current_epoch)
        metrics.setdefault("update", trainer.global_updates)

        line = []
        # Use the stored metric structure (_metrics and _line)
        for entry, fmt in zip(self._metrics, self._line):
            if entry in metrics:
                line.append(fmt)
            else:
                # Missing (e.g., if a metric only appears sometimes)
                width = self._widths.get(entry, self.MIN_COLUMN_WIDTH)

                # Use a string format specifier to center the "[n/a]" placeholder
                # within the calculated column width.
                line.append(f"%({entry}){width}s")
                metrics[entry] = "[n/a]".center(
                    width
                )  # Add the placeholder string to the metrics dictionary

        return " ".join(line), metrics

    @rank_zero.rank_zero_only
    def do_log(self, *args, **kwargs):
        if self.is_enabled:
            _LOGGER.log(self._log_level, *args, **kwargs)


# Helper to check if a value is a scalar (for logging)
def is_scalar(value) -> bool:
    """Checks if a value is a scalar number suitable for logging."""
    # Checks for standard Python numbers, and you might need to add checks
    # for JAX/Numpy/other array types if they aren't standard Python numbers
    # but represent a single number (e.g., jax.Array with shape () or (1,))
    if isinstance(value, numbers.Number):
        return True

    # Example for potential JAX/Numpy scalar check (untested without environment)
    if hasattr(value, "shape") and value.shape == ():
        return True

    return False
