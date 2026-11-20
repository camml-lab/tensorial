"""ReActrix listeners that render parity plots of model outputs.

The :class:`ParityPlotter` and :class:`GraphParityPlotter` listeners
collect true and predicted values across a training stage and render them as
scattered parity plots (``y'`` vs ``y``) using Matplotlib.  The figures are
handed to each of the trainer's loggers via :meth:`reax.Logger.log_figure`, so
each logging backend decides how to store them (e.g. as image files in the log
directory, as TensorBoard images or as MLflow artifacts).

Each plotter collects one or more *series*, each giving its own parity plot.  A series is
identified by a ``(quantity, component)`` key, either of which may be ``None``, and is logged as
``<figure_name>/<quantity>/<stage>/<component>`` (leaving out the parts that are ``None``).
"""

from collections.abc import Mapping, Sequence
from typing import Any, Final

import e3nn_jax as e3j
import jraph
import matplotlib.pyplot as plt
import numpy as np
import reax
from typing_extensions import override

from ... import base, gcnn
from ...gcnn import _tree
from ..utils import pylogger

__all__ = (
    "ParityPlotter",
    "GraphParityPlotter",
    "IrrepsGraphParityPlotter",
)

_LOGGER = pylogger.RankedLogger(__name__, rank_zero_only=True)

STAGES: Final[tuple[str, ...]] = ("train", "validation", "test", "predict")

SeriesKey = tuple[str | None, str | None]
"""Identifies a series of parity data: ``(quantity, component)``, either may be ``None``."""

SeriesData = tuple[list[np.ndarray], list[np.ndarray]]
"""The collected ``(true values, predicted values)`` of a series, one array per batch."""


class ParityPlotter(reax.TrainerListener):
    """
    A TrainerListener that collects true and predicted values to create a
    parity plot at the end of each stage (Train, Validate, Test, Predict).
    """

    COLOR_CFG: Final[list[tuple[str, str, str]]] = [
        ("train", "#b2df8a", "Train"),
        ("validation", "#1f78b4", "Validation"),
        ("test", "#f19428", "Test"),
    ]

    def __init__(
        self,
        figure_name: str = "parity",
        fit_plot_every: int = 10,
        x_label: str = "True Values (y)",
        y_label: str = "Predicted Values (y')",
        plot_final_combined: bool = False,
    ):
        """Create the plotter.

        Args:
            figure_name: The name under which figures are logged, the stage name (and any further
                qualifiers) are appended as ``/`` separated groups, e.g. ``parity/train``.
            fit_plot_every: For ``train``/``validation`` plots, the maximum number of epochs
                between two plots. Higher values mean fewer plots.
            x_label: Label for the x-axis.
            y_label: Label for the y-axis.
            plot_final_combined: After testing, also log a plot combining the train, validation
                and test data.
        """
        # Params
        self._figure_name: Final[str] = figure_name
        self._plot_every: Final[int] = fit_plot_every
        self._x_label: Final[str] = x_label
        self._y_label: Final[str] = y_label
        self.plot_final_combined = plot_final_combined

        # State
        self._last_plotted_epoch: dict[str, int] = {}
        self._warned_no_loggers = False

        # The data collected in each stage: {stage: {series key: (true values, predicted values)}}
        self.data_store: dict[str, dict[SeriesKey, SeriesData]] = {stage: {} for stage in STAGES}

    def _should_collect(self, stage_name: str, epoch: int | None) -> bool:
        if epoch is None:
            return True
        if stage_name in ("test", "predict"):
            return True
        if epoch == 0 or (epoch + 1) % self._plot_every == 0:
            return True
        last = self._last_plotted_epoch.get(stage_name, 0)
        return (epoch - last) >= self._plot_every

    def _epoch(self, trainer: reax.Trainer) -> int | None:
        """Return the current epoch, or :data:`None` when not fitting.

        ``trainer.current_epoch`` is only a meaningful epoch index while a
        :class:`reax.stages.Fit` stage is running; for standalone
        ``validate``/``test``/``predict`` runs there is no epoch to report.
        """
        return trainer.current_epoch if isinstance(trainer.stage, reax.stages.Fit) else None

    @staticmethod
    def _sanity_checking(trainer: reax.Trainer) -> bool:
        """Whether the fit is running its validation sanity check, which isn't plotted."""
        stage = trainer.stage
        return isinstance(stage, reax.stages.Fit) and stage.sanity_checking

    # --- Series: what is plotted, and how it is labelled ---

    def _collect_series(self, batch: Any, outputs: Any) -> dict[SeriesKey, tuple[Any, Any]]:
        """Return the ``(true, predicted)`` values of each series in a batch.

        Subclasses override this to plot more than one series.  Series that are missing from a
        batch are simply not collected for it.
        """
        return {(None, None): self.get_target_predicted(batch, outputs)}

    def _series_labels(self, key: SeriesKey) -> tuple[str, str, str]:
        """Return the ``(x label, y label, description)`` of a series, used in its plots."""
        return self._x_label, self._y_label, "True vs. Predicted"

    def _series_figure_name(self, key: SeriesKey, stage_name: str) -> str:
        """Return the name, below :attr:`figure_name`, that a series is logged under."""
        quantity, component = key
        return "/".join(part for part in (quantity, stage_name, component) if part is not None)

    def _collect_batch_data(self, stage_name: str, outputs: Any | None, batch: Any) -> bool:
        """Collect the true and predicted values of each series in a batch."""
        if outputs is None:
            return False

        store = self.data_store[stage_name]
        for key, (y_true, y_pred) in self._collect_series(batch, outputs).items():
            y_true = np.asarray(base.as_array(y_true)).ravel()
            y_pred = np.asarray(base.as_array(y_pred)).ravel()
            if y_true.size == 0:
                continue  # e.g. everything is masked

            true_list, pred_list = store.setdefault(key, ([], []))
            true_list.append(y_true)
            pred_list.append(y_pred)

        return True

    # --- Plotting ---

    def _scatter_parity(
        self,
        ax: plt.Axes,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        label: str,
        color: Any = None,
        alpha: float = 0.6,
        s: int = 15,
    ):
        ax.scatter(y_true, y_pred, c=color, alpha=alpha, s=s, label=label, edgecolors="none")

    def _finalize_parity_ax(
        self,
        ax: plt.Axes,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        x_label: str,
        y_label: str,
        title: str,
    ):
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        range_buffer = (max_val - min_val) * 0.1
        plot_range = (min_val - range_buffer, max_val + range_buffer)

        ax.plot(
            plot_range, plot_range, color="black", linestyle="--", label="Ideal Parity Line (y=x)"
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.6)

    def _log_figure(self, trainer: reax.Trainer, name: str, figure: plt.Figure) -> None:
        """Log ``figure`` to all the trainer's loggers, and close it.

        ``name`` is appended to :attr:`figure_name`.  During a fit the figure is logged at the
        current update step, for standalone stages without a step.
        """
        try:
            if not trainer.loggers:
                if not self._warned_no_loggers:
                    _LOGGER.warning(
                        "%s: the trainer has no loggers, parity plots will not be saved.",
                        type(self).__name__,
                    )
                    self._warned_no_loggers = True
                return

            step = trainer.global_updates if self._epoch(trainer) is not None else None
            for logger in trainer.loggers:
                logger.log_figure(f"{self._figure_name}/{name}", figure, step)
        finally:
            plt.close(figure)

    def _plot_parity(self, trainer: reax.Trainer, stage_name: str, epoch: int | None = None):
        """Create and log a parity plot of each series collected in a stage."""
        series = self.data_store[stage_name]
        if not series:
            _LOGGER.debug("Skipping parity plot for %s: No data collected.", stage_name)
            return

        if epoch is not None:
            self._last_plotted_epoch[stage_name] = epoch

        for key, (true_list, pred_list) in series.items():
            y_true = np.concatenate(true_list)
            y_pred = np.concatenate(pred_list)
            x_label, y_label, description = self._series_labels(key)

            fig, ax = plt.subplots(figsize=(8, 8))
            self._scatter_parity(ax, y_true, y_pred, label=f"{stage_name} points (N={len(y_true)})")
            self._finalize_parity_ax(
                ax,
                y_true,
                y_pred,
                x_label=x_label,
                y_label=y_label,
                title=f"Parity Plot: {description} ({stage_name.capitalize()})",
            )

            fig.tight_layout()
            self._log_figure(trainer, self._series_figure_name(key, stage_name), fig)

    def _plot_combined_all_stages(self, trainer: reax.Trainer):
        """Plot the train, validation and test data of each series in a single parity plot."""
        stages = [stage for stage, _color, _label in self.COLOR_CFG]
        # All the series, in the order they were first collected
        keys = dict.fromkeys(key for stage in stages for key in self.data_store[stage])

        for key in keys:
            stage_data = {}
            for stage in stages:
                if key in self.data_store[stage]:
                    true_list, pred_list = self.data_store[stage][key]
                    stage_data[stage] = (np.concatenate(true_list), np.concatenate(pred_list))

            x_label, y_label, description = self._series_labels(key)
            self._plot_combined_data(
                trainer,
                stage_data,
                self._series_figure_name(key, "combined"),
                title=f"Global Parity: {description}",
                x_label=x_label,
                y_label=y_label,
            )

    def _plot_combined_data(
        self,
        trainer: reax.Trainer,
        stage_data: dict[str, tuple[np.ndarray, np.ndarray]],
        name: str,
        title: str,
        x_label: str,
        y_label: str,
        label_prefix: str = "",
    ):
        """Core tool to plot multiple stages with consistent colors."""
        fig, ax = plt.subplots(figsize=(7, 6))

        all_true, all_pred = [], []

        # Follow the order and colors in COLOR_CFG
        for stage, color, label in self.COLOR_CFG:
            if stage in stage_data:
                y_true, y_pred = stage_data[stage]
                self._scatter_parity(
                    ax, y_true, y_pred, label=f"{label_prefix}{label}", color=color, alpha=0.5, s=15
                )
                all_true.append(y_true)
                all_pred.append(y_pred)

        if not all_true:
            plt.close(fig)
            return

        self._finalize_parity_ax(
            ax,
            np.concatenate(all_true),
            np.concatenate(all_pred),
            x_label=x_label,
            y_label=y_label,
            title=title,
        )

        fig.tight_layout()
        self._log_figure(trainer, name, fig)

    # --- Implement Batch End Hooks to Collect Data ---

    @override
    def on_train_batch_end(
        self,
        trainer: reax.Trainer,
        _stage: reax.Stage,
        loop: reax.stages.TrainEpoch,
        outputs: Any,
        /,
    ) -> None:
        """Collect true and predicted values at the end of each training batch."""
        if self._should_collect("train", self._epoch(trainer)):
            self._collect_batch_data("train", outputs, loop.batch)

    @override
    def on_validation_batch_end(
        self,
        trainer: reax.Trainer,
        _stage: reax.Stage,
        loop: reax.stages.ValidationEpoch,
        outputs: Any,
        /,
    ) -> None:
        """Collect true and predicted values at the end of each validation batch."""
        if self._sanity_checking(trainer):
            return
        if self._should_collect("validation", self._epoch(trainer)):
            self._collect_batch_data("validation", outputs, loop.batch)

    @override
    def on_test_batch_end(
        self,
        trainer: reax.Trainer,
        _stage: reax.Stage,
        loop: reax.stages.TestEpoch,
        outputs: Any,
        /,
    ) -> None:
        """Collect true and predicted values at the end of each test batch."""
        if self._should_collect("test", self._epoch(trainer)):
            self._collect_batch_data("test", outputs, loop.batch)

    @override
    def on_predict_batch_end(
        self,
        trainer: reax.Trainer,
        _stage: reax.Stage,
        loop: reax.stages.PredictionEpoch,
        outputs: Any,
        /,
    ) -> None:
        """Collect true and predicted values at the end of each predict batch."""
        if self._should_collect("predict", self._epoch(trainer)):
            self._collect_batch_data("predict", outputs, loop.batch)

    # --- Implement Stage End Hooks to Trigger Plotting ---

    @override
    def on_train_epoch_end(
        self, trainer: reax.Trainer, _stage: reax.Stage, _loop: reax.stages.TrainEpoch, /
    ) -> None:
        """A training epoch has ended, plot its data if it was collected."""
        epoch = self._epoch(trainer)
        if epoch is not None and self._should_collect("train", epoch):
            self._plot_parity(trainer, "train", epoch)

    @override
    def on_validation_epoch_end(
        self, trainer: reax.Trainer, _stage: reax.Stage, _loop: reax.stages.ValidationEpoch, /
    ) -> None:
        """A validation epoch within a fit has ended, plot its data if it was collected."""
        epoch = self._epoch(trainer)
        if self._sanity_checking(trainer):
            return
        if epoch is not None and self._should_collect("validation", epoch):
            self._plot_parity(trainer, "validation", epoch)

    @override
    def on_validation_end(self, trainer: reax.Trainer, _stage: reax.Stage, /) -> None:
        """A standalone validation has ended, plot its data."""
        if self._epoch(trainer) is None:
            self._plot_parity(trainer, "validation")

    @override
    def on_test_end(self, trainer: reax.Trainer, _stage: reax.Stage, /) -> None:
        """Test has ended, plot the collected test data and the combined plot if requested."""
        self._plot_parity(trainer, "test", self._epoch(trainer))

        if self.plot_final_combined:
            self._plot_combined_all_stages(trainer)

    @override
    def on_predict_end(self, trainer: reax.Trainer, _stage: reax.Stage, /) -> None:
        """Predict is ending, plot the collected prediction data."""
        self._plot_parity(trainer, "predict", self._epoch(trainer))

    # --- Clear the stored data when a collected epoch starts, so only that epoch is held ---
    # Clearing only on collected epochs keeps the last collected one for the combined plot.

    @override
    def on_train_epoch_start(
        self, trainer: reax.Trainer, _stage: reax.Stage, _loop: reax.stages.TrainEpoch, /
    ) -> None:
        if self._should_collect("train", self._epoch(trainer)):
            self._clear("train")

    @override
    def on_validation_epoch_start(
        self, trainer: reax.Trainer, _stage: reax.Stage, _loop: reax.stages.ValidationEpoch, /
    ) -> None:
        if self._sanity_checking(trainer):
            return
        if self._should_collect("validation", self._epoch(trainer)):
            self._clear("validation")

    @override
    def on_test_start(self, _trainer: reax.Trainer, _stage: reax.stages.Test, /) -> None:
        self._clear("test")

    @override
    def on_predict_start(self, _trainer: reax.Trainer, _stage: reax.stages.Predict, /) -> None:
        self._clear("predict")

    def _clear(self, stage_name: str) -> None:
        """Clear the stored data of a stage."""
        self.data_store[stage_name] = {}

    def get_target_predicted(self, batch, outputs) -> tuple[np.ndarray, np.ndarray]:
        """Return the flattened ``(targets, predictions)`` arrays for a single batch.

        Both arrays are converted with :func:`base.as_array` and returned as
        plain ``numpy.ndarray`` instances so they can be plotted directly.

        Args:
            batch: The input batch as a ``(x, y)`` tuple (or a single element).
            outputs: The model outputs, either a scalar tensor or a
                ``{"targets": ..., "predictions": ...}`` mapping.

        Returns:
            A ``(targets, predictions)`` pair as ``numpy.ndarray``.
        """
        targets, predictions = self._get_target_predicted(batch, outputs)
        return np.array(base.as_array(targets)), np.array(base.as_array(predictions))

    def _get_target_predicted(self, batch, outputs) -> tuple[Any, Any]:
        if isinstance(outputs, dict):
            if isinstance(batch, tuple):
                targets = self._get_batch_targets(batch)
            else:
                targets = outputs["targets"]
            predictions = outputs["predictions"]

            return targets, predictions

        targets = batch[1] if len(batch) > 1 and batch[1] is not None else batch[0]
        predictions = outputs
        return targets, predictions

    def _get_batch_targets(self, batch: tuple) -> Any:
        if len(batch) == 1:
            return batch[0]

        if batch[1] is None:
            return batch[0]

        return batch[1]


class GraphParityPlotter(ParityPlotter):
    """Parity plotter specialised for :class:`tensorial.gcnn.typing.GraphsTuple` models.

    Extends :class:`ParityPlotter` so that it knows how to pull target and predicted values out
    of a ``GraphsTuple`` via ``TreePath``-style selectors (similar to
    :func:`tensorial.gcnn._tree.get`).  One or more quantities can be plotted, each getting its
    own parity plots, logged as ``<figure_name>/<quantity>/<stage>``, where ``<quantity>`` is the
    last node of the target path (e.g. ``parity/energy/train``).

    The same masks as :class:`tensorial.gcnn.Loss` are applied: the padding mask of the
    quantity's level (e.g. ``nodes.mask`` for ``nodes.forces``) and the per-field mask next to
    the target (e.g. ``globals.stress_mask`` for ``globals.stress``, see
    :func:`tensorial.gcnn.keys.mask_for`), when present.

    Example::

        GraphParityPlotter(["globals.energy", "nodes.forces", "globals.stress"], optional=True)
    """

    def __init__(
        self,
        targets: (
            gcnn.typing.TreePathLike
            | Sequence[gcnn.typing.TreePathLike]
            | Mapping[str, gcnn.typing.TreePathLike | None]
        ),
        predictions: gcnn.typing.TreePathLike | None = None,
        figure_name: str = "parity",
        fit_plot_every: int = 100,
        x_label: str | None = None,
        y_label: str | None = None,
        optional: bool = False,
        **kwargs,
    ):
        """Create the plotter.

        Args:
            targets: The quantities to plot, as paths into the target graph.  This is either a
                single path (a string such as ``"globals.energy"``, or a tuple of path nodes),
                a list of paths, or a mapping from target paths to prediction paths (``None``
                for the default prediction path).
            predictions: The path into the prediction graph, for a single target only.  If
                ``None``, the prediction path is derived from the target path by replacing the
                last node with ``tensorial.gcnn.keys.predicted(<last>)``.
            figure_name: The name under which figures are logged, the quantity and stage are
                appended, see :class:`ParityPlotter`.  Defaults to ``"parity"``.
            fit_plot_every: For ``train``/``validation`` plots, the maximum
                number of epochs between two plots. Higher values mean fewer
                plots.
            x_label: Custom label for the x-axis. Defaults to
                ``"True Values (<last target node>)"``.
            y_label: Custom label for the y-axis. Defaults to
                ``"Predicted Values (<last prediction node>)"``.
            optional: If ``True``, quantities that are missing from the target or prediction
                graph (e.g. stresses of molecules) are skipped instead of raising an error.
        """
        super().__init__(
            figure_name=figure_name,
            fit_plot_every=fit_plot_every,
            x_label=x_label or "",
            y_label=y_label or "",
            **kwargs,
        )
        self._custom_x_label = x_label
        self._custom_y_label = y_label
        self._optional = optional
        self._quantities: Final[dict[str, tuple[gcnn.typing.TreePath, gcnn.typing.TreePath]]] = (
            _init_quantities(targets, predictions)
        )

    @override
    def _collect_series(self, batch: Any, outputs: Any) -> dict[SeriesKey, tuple[Any, Any]]:
        targets_graph, predictions_graph = self._get_target_predicted(batch, outputs)

        series: dict[SeriesKey, tuple[Any, Any]] = {}
        for name, (target_path, prediction_path) in self._quantities.items():
            values = self._get_quantity(
                targets_graph, predictions_graph, target_path, prediction_path
            )
            if values is None:
                continue

            for component, component_values in self._decompose(*values).items():
                series[(name, component)] = component_values

        return series

    def _get_quantity(
        self,
        targets_graph: jraph.GraphsTuple,
        predictions_graph: jraph.GraphsTuple,
        target_path: gcnn.typing.TreePath,
        prediction_path: gcnn.typing.TreePath,
    ) -> tuple[Any, Any] | None:
        """Return the masked ``(targets, predictions)`` of a quantity, or ``None`` to skip it."""
        if self._optional and not (
            _tree.has_path(targets_graph, target_path)
            and _tree.has_path(predictions_graph, prediction_path)
        ):
            return None

        targets = _tree.get(targets_graph, target_path)
        predictions = _tree.get(predictions_graph, prediction_path)

        if not base.is_tensorial_array_like(targets):
            raise TypeError(
                f"Expected an array-like or IrrepsArray at "
                f"{_tree.path_to_str(target_path)!r}, got {type(targets).__name__}"
            )
        if not base.is_tensorial_array_like(predictions):
            raise TypeError(
                f"Expected an array-like or IrrepsArray at "
                f"{_tree.path_to_str(prediction_path)!r}, got {type(predictions).__name__}"
            )

        # The padding mask of the quantity's level, and the per-field mask, as used by the losses
        mask = _tree.get_mask(targets_graph, target_path)
        field_mask = _tree.get_field_mask(targets_graph, target_path)
        if field_mask is not None:
            mask = field_mask if mask is None else np.asarray(mask) & np.asarray(field_mask)
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            targets = _apply_mask(targets, mask)
            predictions = _apply_mask(predictions, mask)

        return targets, predictions

    def _decompose(self, targets: Any, predictions: Any) -> dict[str | None, tuple[Any, Any]]:
        """Split a quantity into the components that get their own plots (none by default)."""
        return {None: (targets, predictions)}

    @override
    def _series_labels(self, key: SeriesKey) -> tuple[str, str, str]:
        quantity, component = key
        assert quantity is not None
        target_path, prediction_path = self._quantities[quantity]
        suffix = f" {component}" if component is not None else ""

        x_label = self._custom_x_label or f"True Values ({target_path[-1]}{suffix})"
        y_label = self._custom_y_label or f"Predicted Values ({prediction_path[-1]}{suffix})"
        return x_label, y_label, f"{quantity}{suffix}"


def _init_quantities(
    targets: Any, predictions: gcnn.typing.TreePathLike | None
) -> dict[str, tuple[gcnn.typing.TreePath, gcnn.typing.TreePath]]:
    """Return ``{name: (target path, prediction path)}`` for the quantities to plot."""
    pairs: list[tuple[gcnn.typing.TreePath, gcnn.typing.TreePathLike | None]]
    if isinstance(targets, (str, tuple)):
        # A single path (a tuple is the nodes of one path, not a collection of paths)
        pairs = [(_tree.path_from_str(targets), predictions)]
    elif predictions is not None:
        raise ValueError(
            "`predictions` can only be used with a single target, to plot several quantities with "
            "custom prediction paths pass a {target: prediction} mapping as `targets`"
        )
    elif isinstance(targets, Mapping):
        pairs = [(_tree.path_from_str(target), pred) for target, pred in targets.items()]
    elif isinstance(targets, Sequence):
        pairs = [(_tree.path_from_str(target), None) for target in targets]
    else:
        raise TypeError(
            f"`targets` must be a path, a list of paths or a mapping, got {type(targets).__name__}"
        )

    if not pairs:
        raise ValueError("At least one target must be given")

    resolved = [
        (target, _tree.path_from_str(pred) if pred is not None else _default_prediction(target))
        for target, pred in pairs
    ]

    # Name each quantity after the last node of its target path, or the full path if not unique
    leaves = [target[-1] for target, _ in resolved]
    quantities = {}
    for target, pred in resolved:
        name = target[-1] if leaves.count(target[-1]) == 1 else _tree.path_to_str(target)
        if name in quantities:
            raise ValueError(f"Target {_tree.path_to_str(target)!r} is given more than once")
        quantities[name] = (target, pred)

    return quantities


def _default_prediction(target_path: gcnn.typing.TreePath) -> gcnn.typing.TreePath:
    return target_path[:-1] + (gcnn.keys.predicted(target_path[-1]),)


def _apply_mask(value: Any, mask: np.ndarray) -> Any:
    """Select the unmasked rows of ``value``, keeping ``IrrepsArray``s so their irreps survive."""
    if isinstance(value, e3j.IrrepsArray):
        return value[mask]

    return np.array(base.as_array(value))[mask]


class IrrepsGraphParityPlotter(GraphParityPlotter):
    """
    Decomposes true/predicted data (e.g. IrrepsArrays or Cartesian tensors)
    into constituent properties and generates a parity plot for each, logged as
    ``<figure_name>/<quantity>/<stage>/<irrep>``.
    """

    def _decompose_data(self, data: Any) -> dict[str, np.ndarray]:
        """Decompose data into a dictionary of properties to plot."""
        if isinstance(data, e3j.IrrepsArray):
            # Decompose IrrepsArray into its constituent irrep segments
            results = {}
            for i, ((_mul, ir), segment) in enumerate(zip(data.irreps, data.irreps.slices())):
                # We use the irrep string as the name.
                # If multiple segments have same irrep, we append index.
                name = str(ir)
                if name in results:
                    name = f"{name}_{i}"
                # Slice the segment itself, indexing by irrep would give the last segment of it
                results[name] = np.array(data.array[..., segment])
            return results

        return {"value": np.array(base.as_array(data))}

    @override
    def _decompose(self, targets: Any, predictions: Any) -> dict[str | None, tuple[Any, Any]]:
        decomposed_targets = self._decompose_data(targets)
        decomposed_predictions = self._decompose_data(predictions)
        return {
            name: (values, decomposed_predictions[name])
            for name, values in decomposed_targets.items()
        }
