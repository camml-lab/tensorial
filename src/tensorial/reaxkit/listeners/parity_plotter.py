import pathlib
from typing import Any, Final

import jraph
import matplotlib.pyplot as plt
import numpy as np
import reax
from typing_extensions import override

from ... import base, gcnn
from ...gcnn import _tree
from ..utils import pylogger

__all__ = ("ParityPlotter", "GraphParityPlotter")

_LOGGER = pylogger.RankedLogger(__name__, rank_zero_only=True)


class ParityPlotter(reax.TrainerListener):
    """
    A TrainerListener that collects true and predicted values to create a
    parity plot at the end of each stage (Train, Validate, Test, Predict).
    """

    def __init__(self, save_dir: str | pathlib.Path = "plots/", fit_plot_every: int = 100):
        # Params
        self._save_dir: Final[pathlib.Path] = pathlib.Path(save_dir)
        self._plot_every: Final[int] = fit_plot_every

        # State
        # Stores data for each stage: {'stage_name': (list of true y's, list of predicted y's)}
        self.data_store: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {
            "train": ([], []),
            "validation": ([], []),
            "test": ([], []),
            "predict": ([], []),
        }

    def _collect_batch_data(self, stage_name: str, outputs: Any, batch: Any) -> None:
        """Helper to collect true (y) and predicted (y') values."""
        # Assuming batch is a tuple (x, y) and outputs is y' (the prediction).
        # We take the second element of the batch tuple for the true value (y).
        y_true, y_pred = self.get_target_predicted(batch, outputs)

        # Flatten y_true and y_pred if they are multi-dimensional (e.g., shape (batch_size, 1))
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        self.data_store[stage_name][0].append(y_true)
        self.data_store[stage_name][1].append(y_pred)

    def _plot_parity(self, stage_name: str, save_dir: pathlib.Path):
        """Helper to create and display the parity plot for the current data."""
        true_y_list, pred_y_list = self.data_store[stage_name]

        if not true_y_list:
            _LOGGER.debug("Skipping parity plot for %s: No data collected.", stage_name)
            return

        # 1. Combine all batches into one large numpy array
        # The user requested one large output array which is passed to matplotlib.
        y_true_all = np.concatenate(true_y_list)
        y_pred_all = np.concatenate(pred_y_list)

        # 2. Clear the stage data for the next run (e.g., next 'fit' call)
        self.data_store[stage_name] = ([], [])

        # 3. Create the Parity Plot
        _LOGGER.debug("Generating Parity Plot for %s stage...", stage_name)

        plt.figure(figsize=(8, 8))

        # Determine the range for the ideal line (y=x)
        min_val = min(y_true_all.min(), y_pred_all.min())
        max_val = max(y_true_all.max(), y_pred_all.max())

        # Add a buffer for visualization
        range_buffer = (max_val - min_val) * 0.1
        plot_range = (min_val - range_buffer, max_val + range_buffer)

        # Scatter plot of True vs. Predicted
        plt.scatter(
            y_true_all, y_pred_all, alpha=0.6, label=f"{stage_name} points (N={len(y_true_all)})"
        )

        # Ideal Line (y = x)
        plt.plot(
            plot_range, plot_range, color="red", linestyle="--", label="Ideal Parity Line (y=x)"
        )

        plt.xlabel("True Values (y)")
        plt.ylabel("Predicted Values (y')")
        plt.title(f"Parity Plot: True vs. Predicted ({stage_name.capitalize()} Stage)")
        plt.legend()
        plt.grid(True)

        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_dir / f"{stage_name}.pdf"), bbox_inches="tight")

        # Triggering an image for context/illustration
        _LOGGER.debug("Parity Plot for %s generated.", stage_name)

    def reset(self):
        self.data_store: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {
            "train": ([], []),
            "validation": ([], []),
            "test": ([], []),
            "predict": ([], []),
        }

    # --- Implement Batch End Hooks to Collect Data ---

    @override
    def on_train_batch_end(
        self,
        trainer: reax.Trainer,
        stage: reax.stages.Train,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        /,
    ) -> None:
        self._collect_batch_data("train", outputs, batch)

    @override
    def on_validation_batch_end(
        self,
        trainer: reax.Trainer,
        stage: reax.stages.Validate,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        /,
    ) -> None:
        self._collect_batch_data("validation", outputs, batch)

    @override
    def on_test_batch_end(
        self,
        trainer: reax.Trainer,
        stage: reax.stages.Test,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        /,
    ) -> None:
        self._collect_batch_data("test", outputs, batch)

    @override
    def on_predict_batch_end(
        self,
        trainer: reax.Trainer,
        stage: reax.stages.Predict,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        /,
    ) -> None:
        self._collect_batch_data("predict", outputs, batch)

        # --- Implement Stage End Hooks to Trigger Plotting ---

    @override
    def on_train_end(self, trainer: reax.Trainer, stage: reax.stages.Train, /):
        """Training is ending, plot the collected training data."""
        if (stage.epoch - 1) % self._plot_every == 0:
            self._plot_parity("train", self._get_save_dir(trainer))

    @override
    def on_validation_end(self, trainer: reax.Trainer, stage: reax.stages.Validate, /) -> None:
        """Validation has ended, plot the collected validation data."""
        if (stage.epoch - 1) % self._plot_every == 0:
            self._plot_parity("validation", self._get_save_dir(trainer))

    @override
    def on_fit_end(self, trainer: "reax.Trainer", stage: "reax.stages.Fit", /) -> None:
        """Fit has ended, plot the collected training data."""
        self._plot_parity("train", self._get_save_dir(trainer))
        self._plot_parity("validation", self._get_save_dir(trainer))

    @override
    def on_test_end(self, trainer: reax.Trainer, stage: reax.stages.Test, /) -> None:
        """Test has ended, plot the collected test data."""
        self._plot_parity("test", self._get_save_dir(trainer))

    @override
    def on_predict_end(self, trainer: reax.Trainer, stage: reax.stages.Predict, /) -> None:
        """Predict is ending, plot the collected prediction data."""
        self._plot_parity("predict", self._get_save_dir(trainer))

    def get_target_predicted(self, batch, outputs) -> tuple[np.ndarray, np.ndarray]:
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

        targets = batch[1] if batch[1] is not None else batch[0]
        predictions = outputs
        return targets, predictions

    def _get_batch_targets(self, batch: tuple) -> Any:
        if len(batch) == 1:
            return batch[0]

        if batch[1] is None:
            return batch[0]

        return batch[1]

    def _get_save_dir(self, trainer: reax.Trainer) -> pathlib.Path:
        if self._save_dir.is_absolute():
            return self._save_dir

        return trainer.log_dir / self._save_dir


class GraphParityPlotter(ParityPlotter):
    def __init__(
        self,
        targets: gcnn.typing.TreePathLike,
        predictions: gcnn.typing.TreePathLike | None = None,
        save_dir: str | pathlib.Path = "plots/",
    ):
        super().__init__(save_dir)
        self._target_path = gcnn.utils.path_from_str(targets)
        self._prediction_path = self._init_prediction_path(predictions, self._target_path)

    @staticmethod
    def _init_prediction_path(
        prediction_path, target_path: gcnn.typing.TreePath
    ) -> gcnn.typing.TreePath:
        if prediction_path is not None:
            return prediction_path

        path = list(target_path)
        path[-1] = gcnn.keys.predicted(path[-1])
        return tuple(path)

    @override
    def _get_target_predicted(
        self, batch: tuple[jraph.GraphsTuple, jraph.GraphsTuple | None], outputs: jraph.GraphsTuple
    ) -> tuple[Any, Any]:
        targets_graph, predictions_graph = super()._get_target_predicted(batch, outputs)

        targets = _tree.get(targets_graph, self._target_path)
        predictions = _tree.get(predictions_graph, self._prediction_path)

        mask_path = self._target_path[:-1] + ("mask",)
        try:
            mask = np.array(_tree.get(targets_graph, mask_path))
        except KeyError:
            pass
        else:
            targets = np.array(base.as_array(targets))
            predictions = np.array(base.as_array(predictions))
            targets = targets[mask]
            predictions = predictions[mask]

        return targets, predictions
