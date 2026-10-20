Profiling: finding out where the time goes
==========================================

Training a model can take hours or days. *Profiling* records what the
computer is doing during a few training steps, so you can see where the time is
spent and why a run is slower than you expected. ``tensorial`` can do this for
you with a single extra option on the command line.

You do not need to understand how profiling works to use it: add
``--profile`` to your command, wait for the run, and open the file it produces.

.. contents::
    :local:

When is it useful?
------------------

- A run is much slower than you expected.
- You are about to move from a CPU to a GPU, or from one GPU to several, and
  want to check that the extra hardware is actually being used.
- You changed the model or the dataset and want to know whether it got slower.

If your runs are already fast enough, you do not need profiling.

Quick start
-----------

Run the command you normally use for training, and add ``--profile``:

.. code-block:: shell

    tensorial train -i train.yaml --profile

As usual, run it from the folder that contains ``train.yaml`` (if you do not
have one yet, see :ref:`profiling-example-folder`).

Here is what happens:

1. The first 3 training steps run normally. They are slow because the code is
   being prepared for your hardware (this is called *compilation*), which is
   not what you want to measure, so they are not recorded.
2. The next 10 steps are recorded.
3. The recording is saved and training goes on as usual.

When the recording is saved, the log shows a message like this (the folder name
will be different on your machine)::

    Profile trace of 10 batches saved to logs/train/2026-10-12_09-30-00/profile.

From that moment the file is on disk. You can wait for the training to finish or
stop it with ``Ctrl+C``.

.. _profiling-example-folder:

If you do not have a ``train.yaml`` yet
---------------------------------------

The ``train.yaml`` file only works together with the other configuration
folders next to it (``model``, ``trainer``, ``hydra``, ...). To get a complete
set to start from, copy the example that comes with ``tensorial``:

.. code-block:: shell

    tensorial examples init mlip.si my_experiment
    cd my_experiment/mlip-si

Then put your data file in the ``data`` folder, as described in the
``README.rst`` of the example. This command only copies configuration files: it
does not run anything.

Options
-------

All the profiling options start with ``--profile`` and can be used with
``tensorial train`` and ``tensorial predict``. You can put them anywhere in the
command, together with the usual settings such as ``trainer.devices=2``.

.. list-table::
    :header-rows: 1
    :widths: 28 14 58

    * - Option
      - Default
      - What it does
    * - ``--profile``
      -
      - Turn profiling on.
    * - ``--profile-dir DIR``
      - the run's log folder
      - Where to save the recording. Using this option also turns profiling on,
        so ``--profile`` is not needed. By default the recording goes into a
        ``profile`` folder inside the folder of the run.
    * - ``--profile-warmup N``
      - 3
      - How many steps to run *before* recording starts. Leave this alone,
        unless the warning described in :ref:`profiling-troubleshooting` appears.
    * - ``--profile-steps N``
      - 10
      - How many steps to record. A small number is better: recordings grow
        quickly (see :ref:`profiling-troubleshooting`).
    * - ``--profile-no-python``
      -
      - Leave out the details of the Python code. The recording becomes much
        smaller and you still see what the hardware does. Using this option also
        turns profiling on.

Examples
--------

Choose where the recording is saved:

.. code-block:: shell

    tensorial train -i train.yaml --profile-dir my_profile

Record 5 steps after 5 steps of warm-up, and keep the recording small:

.. code-block:: shell

    tensorial train -i train.yaml --profile --profile-warmup 5 --profile-steps 5 --profile-no-python

Profile on two GPUs (see the note below):

.. code-block:: shell

    tensorial train -i train.yaml trainer=gpu trainer.accelerator=cuda trainer.devices=2 --profile --profile-no-python

Profile a short test run, instead of waiting for a whole training. The settings
at the end of the line (they are the usual ones, not profiling options) make the
run stop after two short epochs of at most 20 steps each, which is enough for the
3 warm-up steps and the 10 recorded steps:

.. code-block:: shell

    tensorial train -i train.yaml --profile train.max_epochs=2 +train.limit_train_batches=20

Profile the evaluation of a trained model instead of training (the options are
the same):

.. code-block:: shell

    tensorial predict -i eval.yaml --profile

.. note::

    When several processes are used (for example ``trainer.devices=2``), each one
    writes its own recording in a folder called ``process_0``, ``process_1``,
    and so on, inside the profile folder.

Where is the recording?
-----------------------

Inside the profile folder, in ``plugins/profile/<date and time>/``:

.. code-block:: text

    logs/train/<date>_<time>/profile/plugins/profile/<date and time>/
        perfetto_trace.json.gz    <- open this one in the Perfetto website
        vm.trace.json.gz
        vm.xplane.pb              <- used by xprof (see below)

On a cluster, run the profiling inside your usual job, then copy the
``profile`` folder to your own computer to look at it, for example with
``scp -r`` or ``rsync -a``.

How to look at it
-----------------

**In your web browser (the easiest way).** Go to https://ui.perfetto.dev, click
*Open trace file* in the left menu (or drag the file onto the page) and choose
``perfetto_trace.json.gz``. You will see a timeline: use the ``W`` and ``S``
keys to zoom in and out, and ``A`` and ``D`` to move left and right. Every
training step appears as a block named ``train_step`` (``test_step`` when
profiling ``predict``).

.. figure:: images/perfetto_gpu_trace.png
    :alt: A GPU training trace opened in Perfetto: the GPU rows at the top, with
          short bursts of activity separated by long empty gaps, and the CPU rows
          below.
    :width: 100%

    About 240 ms of a training run on one GPU (QM9 dataset, batches of 128
    structures), opened in Perfetto. The whole recording lasts 938 ms.

The figure above was made with this command:

.. code-block:: shell

    tensorial train -i train.yaml trainer=gpu trainer.accelerator=cuda trainer.devices=1 \
        --profile-dir test1 '~model.metrics.frmse' data.batch_size=128 \
        train.max_epochs=50 +trainer.enable_checkpointing=false

Here is how to read it, from top to bottom (the times are approximate, read from
the ruler at the top of the timeline):

1. **The GPU** (``device:GPU:0``). The first row, ``Stream #14 (MemcpyD2D,
   Compute)``, is where the GPU does its calculations. Each coloured block is a
   short burst of about 7 ms; the thin coloured stripes inside it are the many
   small calculations that make up the model. A burst comes roughly every 30 ms,
   so a training step (about 60 ms here) contains **two bursts**: one computes the
   model's predictions (energies), the other computes how to change the model to
   reduce its error (the *gradients*).
2. **The gaps on the GPU row.** Between two bursts, the GPU row is empty for more
   than 20 ms: the GPU has nothing to do and waits for the processor (CPU) to send
   it the next piece of work. In this example the GPU computes only about a
   quarter of the time. Large gaps like these mean that the GPU is not fully used:
   a bigger model or bigger batches would cost little extra time.
3. **Copies between the computer's memory and the GPU.** ``Stream #15 (MemcpyH2D)``
   shows the data being sent *to* the GPU (the next batch), and ``Stream #16`` to
   ``#19 (MemcpyD2H)`` show results being sent *back* from the GPU (for example the
   values of the loss that are written in the log). They happen once per step,
   about every 60 ms, and the marks are very thin: moving the data is not what
   takes time here.
4. **The processor** (``host:CPU``). The ``Host Threads`` and
   ``pjrt_async_work_runner`` rows show the CPU preparing and launching the work.
   Their marks are grouped at the same moments as the copies: this is when a new
   step starts. The rest of the time is spent in Python code, which is hidden in
   this view.

.. tip::

    This recording was made *with* the details of the Python code, which slows
    the Python part down a little. To check that the gaps are not caused by the
    recording itself, record again with ``--profile-no-python`` and compare.

**With xprof.** If you prefer the TensorBoard-style views, install it and point
it at the profile folder:

.. code-block:: shell

    pip install xprof
    xprof my_profile

**Only the numbers, without a graphical interface.** This is handy on a cluster.
Replace ``<date and time>`` by the name of the folder inside
``plugins/profile`` and ``my_profile`` by your profile folder:

.. code-block:: shell

    xprof aggregate_xplane_events <date and time> --logdir my_profile -e '^train_step$'

It prints how many steps were recorded and their average, shortest and longest
duration (in picoseconds).

What to look for
----------------

You do not need to understand every line of the timeline. Two questions are
usually enough:

- **How long is a step?** Compare the length of the ``train_step`` blocks before
  and after a change (a different model, another batch size, more GPUs...). If
  the steps are not shorter, the change did not help.
- **Is the hardware busy, or waiting?** Long uninterrupted blocks of activity
  mean that the computer is computing. Large empty gaps between the blocks
  usually mean that it is waiting for something else, for example for data to
  be prepared.

If you are not sure how to read a recording, send it, together with the command
you used, to the person who maintains the code on your cluster, or open an issue
on the project's page.

Using a configuration file instead
----------------------------------

If you always want profiling in a given experiment, you can write it in your
``train.yaml`` instead of typing the options. Add this to the ``listeners``
section:

.. code-block:: yaml

    listeners:
      profile_trace:
        _target_: tensorial.reaxkit.listeners.ProfileTrace
        warmup: 3
        steps: 10
        # log_dir: my_profile        # optional, where to save the recording
        # python_tracer_level: 0     # optional, same as --profile-no-python

.. _profiling-troubleshooting:

Something went wrong?
---------------------

.. list-table::
    :header-rows: 1
    :widths: 40 60

    * - What you see
      - What to do
    * - ``Cannot find primary config 'train'`` or ``Could not find 'hydra/default'``
      - You are not in the right folder. Go to the folder that contains
        ``train.yaml`` *and* the ``hydra``, ``model``, ``trainer``... folders (see
        :ref:`profiling-example-folder`).
    * - No ``profile`` folder, and the log says ``nothing was recorded``
      - The run was too short: it ended before the 3 warm-up steps were done. Run
        for longer, or use a smaller ``--profile-warmup``.
    * - The log says ``Training finished after only ... batches to profile, saving
        the partial trace``
      - The run ended before all the requested steps were recorded. The
        recording was saved anyway, with fewer steps. Nothing is lost.
    * - ``Could not start the profile trace: Profile has already been started``
      - Another profiler is already running. Do not use ``--profile`` together
        with ``profiler=jax`` in the same command: the second one cannot start.
    * - The recording is very large, or the run stops with a message about ``2GB``
      - Record fewer steps (a small ``--profile-steps``), add
        ``--profile-no-python``, and do not profile a whole training: the limit
        for one recording is 2 GB.
    * - ``error: --profile-steps must be >= 1`` (or ``--profile-warmup must be
        >= 0``)
      - One of the values is not valid. Use a whole number.

.. note::

    The recordings made with ``--profile`` show the steady state of the
    training. If the shape of your data changes a lot from one step to the next,
    the code can be prepared again for each new shape (compilation) during the
    recording, which makes the timings look slower than they really are. In that
    case, increase ``--profile-warmup``.

Further reading
---------------

- :doc:`mlip_guide` for building and training interatomic potentials.
- The `JAX profiling documentation <https://docs.jax.dev/en/latest/profiling.html>`_
  for the tools behind ``--profile``.
