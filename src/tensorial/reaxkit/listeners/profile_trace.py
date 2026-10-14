"""A listener that records a short JAX profiler trace after the JIT warm-up.

Tracing a whole run is rarely useful (it mostly records compilation) and quickly becomes huge:
the profiler cannot serialise a trace larger than 2 GB.  Instead, this listener lets a number of
batches run un-traced so that everything is compiled, and then records the next few batches::

    # warm up JIT first: ``warmup`` batches run untraced
    # then record ``steps`` batches
    with jax.profiler.trace(log_dir):
        ...

It is normally enabled from the command line with ``tensorial train --profile`` (or
``tensorial predict --profile``), see :func:`tensorial.reaxkit.cli.main_cli`.  The resulting
trace can be opened with `xprof <https://github.com/openxla/xprof>`_ / TensorBoard or, since
``perfetto_trace.json.gz`` is written by default, dragged into https://ui.perfetto.dev.
"""

from collections.abc import Sequence
import logging
import os
import pathlib
from typing import Any, Final

import jax
import reax

__all__ = ("ProfileTrace",)

_LOGGER = logging.getLogger(__name__)

TRAIN: Final[str] = "train"
VALIDATION: Final[str] = "validation"
TEST: Final[str] = "test"
PREDICT: Final[str] = "predict"
STAGES: Final[tuple[str, ...]] = (TRAIN, VALIDATION, TEST, PREDICT)
DEFAULT_SUBDIR: Final[str] = "profile"


class ProfileTrace(reax.TrainerListener):
    """Record a JAX profiler trace of a few batches once the JIT warm-up is done.

    Batches of the selected ``stages`` are counted across epochs.  The first ``warmup`` of them
    run untraced; the following ``steps`` are recorded, each one wrapped in a
    :class:`jax.profiler.StepTraceAnnotation` and followed by a ``block_until_ready`` so that the
    asynchronous device work lands inside the trace.  Only one window is ever recorded.

    If the run ends (or fails) while the window is open the trace is stopped and saved, and if the
    run ends before the window starts a warning is logged.

    Args:
        log_dir: The directory to write the trace to.  If ``None`` this will be a ``profile``
            folder inside the trainer's log directory.  When running on several processes, each
            process writes to its own ``process_<i>`` sub-directory.
        warmup: The number of batches to run untraced first.  This must be enough to trigger
            every compilation (the first one or two training steps usually do).
        steps: The number of batches to record.  Keep this small, traces grow quickly.
        stages: The kinds of batches to count and record, any of ``"train"``, ``"validation"``,
            ``"test"`` and ``"predict"``.
        perfetto: Also write ``perfetto_trace.json.gz``, which can be loaded in
            https://ui.perfetto.dev.
        python_tracer_level: The level of the python tracer, see
            :class:`jax.profiler.ProfileOptions`.  Use ``0`` to leave python frames out of the
            trace (much smaller, but you lose the host-side python call stacks).  ``None`` keeps
            the JAX default.
    """

    def __init__(
        self,
        log_dir: str | os.PathLike | None = None,
        warmup: int = 3,
        steps: int = 10,
        stages: Sequence[str] = (TRAIN,),
        perfetto: bool = True,
        python_tracer_level: int | None = None,
    ):
        if warmup < 0:
            raise ValueError(f"warmup must be >= 0, got {warmup}")
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        stages = tuple(str(stage) for stage in stages)
        if not stages or not set(stages) <= set(STAGES):
            raise ValueError(f"stages must be a non-empty subset of {STAGES}, got {stages}")

        self._log_dir = None if log_dir is None else pathlib.Path(log_dir).expanduser()
        self._warmup = warmup
        self._steps = steps
        self._stages = stages
        self._perfetto = perfetto
        self._python_tracer_level = python_tracer_level

        self._seen = 0  # batches finished so far (traced or not)
        self._recorded = 0  # batches recorded in the trace
        self._active = False
        self._done = False
        self._annotation: jax.profiler.StepTraceAnnotation | None = None
        self._trace_dir: pathlib.Path | None = None

    @property
    def trace_dir(self) -> pathlib.Path | None:
        """The directory the trace was (or is being) written to, ``None`` if it never started."""
        return self._trace_dir

    # ----------------------------------------------------------------------------------------
    # Batch hooks
    # ----------------------------------------------------------------------------------------
    def on_train_batch_start(self, trainer, stage, batch, batch_idx, /):
        self._batch_start(trainer, TRAIN)

    def on_train_batch_end(self, trainer, stage, outputs, batch, batch_idx, /):
        self._batch_end(TRAIN, outputs)

    def on_validation_batch_start(self, trainer, stage, batch, batch_idx, /):
        self._batch_start(trainer, VALIDATION)

    def on_validation_batch_end(self, trainer, stage, outputs, batch, batch_idx, /):
        self._batch_end(VALIDATION, outputs)

    def on_test_batch_start(self, trainer, stage, batch, batch_idx, /):
        self._batch_start(trainer, TEST)

    def on_test_batch_end(self, trainer, stage, outputs, batch, batch_idx, /):
        self._batch_end(TEST, outputs)

    def on_predict_batch_start(self, trainer, stage, batch, batch_idx, /):
        self._batch_start(trainer, PREDICT)

    def on_predict_batch_end(self, trainer, stage, outputs, batch, batch_idx, /):
        self._batch_end(PREDICT, outputs)

    # ----------------------------------------------------------------------------------------
    # Run boundaries
    # ----------------------------------------------------------------------------------------
    def on_fit_end(self, trainer, stage, /):
        if TRAIN not in self._stages:
            return
        if self._active:
            _LOGGER.warning(
                "Training finished after only %i of the %i batches to profile, saving the "
                "partial trace",
                self._recorded,
                self._steps,
            )
            self._stop()
        elif not self._done:
            _LOGGER.warning(
                "Profiling was requested but training only ran %i batches and the trace starts "
                "after %i warm-up batches, so nothing was recorded.  Use fewer warm-up batches "
                "or train for longer.",
                self._seen,
                self._warmup,
            )

    def on_stage_end(self, trainer, stage, /):
        # When training, the window can span several epochs, so only the end of the fit
        # closes it.  Otherwise (evaluation, prediction) a stage ending closes the window.
        if TRAIN in self._stages or self._done:
            return
        if self._active:
            _LOGGER.warning(
                "Run finished after only %i of the %i batches to profile, saving the partial "
                "trace",
                self._recorded,
                self._steps,
            )
            self._stop()
        else:
            _LOGGER.info(
                "Profile trace not started yet (%i of %i warm-up batches seen)",
                self._seen,
                self._warmup,
            )

    def on_exception(self, trainer, stage, exception, /):
        if self._active:
            _LOGGER.warning("Run failed, saving the partial profile trace")
            self._stop()

    # ----------------------------------------------------------------------------------------
    # Implementation
    # ----------------------------------------------------------------------------------------
    def _batch_start(self, trainer: "reax.Trainer", kind: str) -> None:
        if self._done or kind not in self._stages:
            return
        if not self._active and self._seen >= self._warmup:
            self._start(trainer)
        if self._active:
            self._annotation = jax.profiler.StepTraceAnnotation(
                f"{kind}_step", step_num=self._recorded
            )
            self._annotation.__enter__()

    def _batch_end(self, kind: str, outputs: Any) -> None:
        if self._done or kind not in self._stages:
            return
        self._seen += 1
        if not self._active:
            return
        # Make sure the (asynchronous) device work of this batch is inside the trace
        jax.block_until_ready(outputs)
        self._close_annotation()
        self._recorded += 1
        if self._recorded >= self._steps:
            self._stop()

    def _resolve_dir(self, trainer: "reax.Trainer") -> pathlib.Path:
        path = self._log_dir
        if path is None:
            path = pathlib.Path(trainer.log_dir) / DEFAULT_SUBDIR
        if jax.process_count() > 1:
            # Processes would otherwise overwrite each other's trace
            path = path / f"process_{jax.process_index()}"
        return path

    def _start(self, trainer: "reax.Trainer") -> None:
        self._trace_dir = self._resolve_dir(trainer)
        kwargs: dict[str, Any] = {}
        if self._perfetto:
            kwargs["create_perfetto_trace"] = True
        if self._python_tracer_level is not None:
            options = jax.profiler.ProfileOptions()
            options.python_tracer_level = self._python_tracer_level
            kwargs["profiler_options"] = options

        _LOGGER.info(
            "Warm-up done (%i batches), recording %i batches to %s",
            self._seen,
            self._steps,
            self._trace_dir,
        )
        try:
            jax.profiler.start_trace(str(self._trace_dir), **kwargs)
        except RuntimeError as exc:  # e.g. another profiler is already running
            _LOGGER.error("Could not start the profile trace: %s", exc)
            self._done = True
            return
        self._active = True

    def _close_annotation(self) -> None:
        if self._annotation is not None:
            self._annotation.__exit__(None, None, None)
            self._annotation = None

    def _stop(self) -> None:
        self._close_annotation()
        try:
            jax.profiler.stop_trace()
        finally:
            self._active = False
            self._done = True
        _LOGGER.info(
            "Profile trace of %i batches saved to %s.  View it with `xprof %s`, or load "
            "perfetto_trace.json.gz in https://ui.perfetto.dev",
            self._recorded,
            self._trace_dir,
            self._trace_dir,
        )
