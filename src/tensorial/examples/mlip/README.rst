si: E(3)-equivariant silicon MLIP example
=========================================

An E(3)-equivariant interatomic potential for silicon, trained on the
``sitraj.xyz`` atomic trajectory (per-atom energies, forces and stresses)
with a NequIP-style backbone.

Data
----
The default configuration reads ``${paths.data_dir}/sitraj.xyz``
(``./data/sitraj.xyz`` relative to where you run), so place that file in a
``data/`` directory next to your working directory, or point at another
file on the command line::

    data.dataloader.path=/path/to/trajectory.extxyz

Run
---
To train::

    tensorial train -i train.yaml

Evaluation expects a trained checkpoint::

    tensorial predict -i eval.yaml ckpt_path=/path/to/checkpoint

You can select an alternative model or override any option on the
command line, e.g. ``tensorial train -i train.yaml model=mace`` or
``data=ase ase_data=/path/to/file.xyz``.

Profiling
---------
To record a `JAX profiler <https://docs.jax.dev/en/latest/profiling.html>`_
trace, add ``--profile`` to ``train`` (or ``predict``)::

    tensorial train -i train.yaml --profile

The first few steps run untraced so that everything is compiled (JIT warm-up),
then a short window of steps is recorded, so the trace is small and shows the
steady state rather than the compilation.  The trace is written to a
``profile`` folder inside the run's output directory (one ``process_<i>``
sub-folder per process when training on several devices).  Open it with
`xprof <https://github.com/openxla/xprof>`_ / TensorBoard, or load the
``perfetto_trace.json.gz`` file in https://ui.perfetto.dev.

Options:

* ``--profile-dir DIR``: write the trace to ``DIR`` (implies ``--profile``)
* ``--profile-warmup N``: steps to run before recording starts (default 3)
* ``--profile-steps N``: steps to record (default 10); keep this small, traces
  grow quickly and the profiler cannot save a trace larger than 2 GB
* ``--profile-no-python``: leave python frames out of the trace, which makes it
  much smaller but removes the host-side python call stacks

For example::

    tensorial train -i train.yaml --profile-dir /tmp/jax_trace --profile-steps 5
