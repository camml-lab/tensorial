Command-line interface
======================

Installing ``tensorial`` puts a ``tensorial`` command on your ``PATH``
(entry point :func:`tensorial.reaxkit.cli.main_cli`). It is a thin wrapper: it
dispatches to the Hydra-driven training and evaluation entry points, or lists,
describes and copies the bundled example configurations so you can run one
straight away.

.. contents::
   :local:

.. code-block:: shell

   tensorial <command> [options]

The available commands are:

- ``train`` — fit a model from a Hydra config.
- ``predict`` — evaluate a trained checkpoint.
- ``examples`` — list, describe or copy the bundled example configurations.

``tensorial train``
-------------------

Train a model described by a Hydra config file. ``-i`` / ``--input`` points at
the config; it defaults to ``configs/train.yaml``:

.. code-block:: shell

   tensorial train -i train.yaml
   tensorial train -i configs/my_exp.yaml

Within the config's directory, Hydra composes the file as the primary config,
so any other YAML groups next to it (``data/``, ``model/``, ``trainer/``, ...)
can be selected or overridden on the command line exactly as with a normal
Hydra app:

.. code-block:: shell

   tensorial train -i train.yaml model=mace data=qm9
   tensorial train -i train.yaml train.max_epochs=50

``-i`` is expected to be a file. If it is a directory, ``predict`` resolves the
default ``config.yaml`` inside it (see below); ``train`` does not.

``tensorial predict``
---------------------

Evaluate a trained model. It takes the same ``-i`` / ``--input`` option,
defaulting to ``configs/eval.yaml``, and forwards everything else to Hydra:

.. code-block:: shell

   tensorial predict -i eval.yaml ckpt_path=/path/to/checkpoint

Unlike ``train``, ``-i`` may also be a *directory*, in which case the file
:data:`tensorial.reaxkit.config.DEFAULT_CONFIG_FILE` (``config.yaml``) is loaded
from it:

.. code-block:: shell

   tensorial predict -i checkpoints/my_run/

An evaluation config normally needs a checkpoint path, e.g. via the
``ckpt_path`` override above.

``tensorial examples``
----------------------

The package bundles ready-to-run example configurations (currently the
``mlip`` example, a machine-learning interatomic potential). This command lists
them, describes them, and copies one into your working directory so you can
run it unmodified.

.. code-block:: shell

   tensorial examples [list [-l|--long] [--json]]   # list the examples
   tensorial examples <name>                        # describe one example
   tensorial examples init <name> [<dest>]          # copy one example

Listing
~~~~~~~

List the available examples, each with a one-line summary of its data:

.. code-block:: shell

   tensorial examples list
   tensorial examples                 # 'list' is the default
   tensorial examples list -l         # also show each example's full description
   tensorial examples list --json     # machine-readable list of names, e.g. for CI
   tensorial examples list -l --json  # names, summaries and descriptions as JSON

which prints:

.. code-block:: text

   Available examples:
     - mlip.3bpa   3BPA drug-like molecule, 500 configs at 300 K (50 for validation); energy, forces
     - mlip.mad    MAD, ~100k diverse structures across most of the periodic table; energy, forces, stress
     - mlip.qm9    QM9 small organic molecules, 10k-molecule subset by default; energy (U0)
     - mlip.rmd17  rMD17 small molecules (aspirin by default), 1000 configs (50 for validation); energy, forces
     - mlip.si     Bulk silicon trajectory, 110 periodic 64-atom structures; energy, forces, stress

   Use 'tensorial examples list --long' for full descriptions.

Describing
~~~~~~~~~~

Give an example's name to show its full description: where its data comes
from, how it is split, and any options worth knowing about. This only prints;
nothing is copied.

.. code-block:: shell

   tensorial examples mlip.qm9

.. code-block:: text

   mlip.qm9 — QM9 small organic molecules, 10k-molecule subset by default; energy (U0)

     QM9 (Ramakrishnan et al., 2014): ~134k small organic molecules (up to 9 heavy atoms of
     ...

   To copy it: tensorial examples init mlip.qm9 [dest]

Giving the name of an example with variants (e.g. ``tensorial examples mlip``)
lists its variants and says which one ``init`` copies by default.

Copying
~~~~~~~

Copy an example into the current directory, or a destination you give:

.. code-block:: shell

   tensorial examples init mlip.si
   tensorial examples init mlip.si path/to/dest

This creates an ``mlip-si/`` directory containing ``train.yaml``,
``eval.yaml`` and the other config groups, plus ``data/si.yaml`` and the
``data/sitraj.xyz`` it reads. From there:

.. code-block:: shell

   cd mlip-si
   tensorial train -i train.yaml

.. note::

   ``init`` is required to copy an example: ``tensorial examples mlip.si``
   only shows its description.

Variants
~~~~~~~~

An example can expose **variants** through its ``data`` Hydra group: the
example differs from its siblings only in which dataset it trains on. The
``mlip`` example ships ``3bpa``, ``mad``, ``qm9``, ``rmd17`` and ``si``
variants, which is why it is listed as ``mlip.3bpa``, ``mlip.mad`` and so on.

Copying a variant (``mlip.si``) rewrites the ``data`` default in the copied
``train.yaml`` and ``eval.yaml`` to that choice, and copies only that variant's
dataset files, not those of its siblings. A bare name (``mlip``) copies the
example with its default ``data`` choice instead.

Writing an example's metadata
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each variant file (or ``train.yaml``, for an example without variants)
describes itself and the files it needs in the comments at its top:

.. code-block:: yaml

   # summary: Bulk silicon trajectory, 110 periodic 64-atom structures; energy, forces, stress
   # requires: sitraj.xyz
   # A short trajectory of bulk silicon (64-atom cubic cell), bundled with the
   # example so it runs offline.  Randomly split 80/10/10 into train/validation/test.
   _target_: tensorial.gcnn.data.ase_graph_module_random_split
   path: ${paths.data_dir}/sitraj.xyz

- ``# summary:`` is the one-line summary shown next to the name by ``list``.
  It is free-form, but should give an idea of how much data there is and which
  labels (energy, forces, stress, ...) the model trains on.
- ``# requires:`` lists files, relative to the variant file, that ``init``
  copies along with it (here, the bundled dataset).
- The **description**, shown by ``tensorial examples <name>`` and
  ``list --long``, is the rest of the comment block at the top of the file, up
  to its first non-comment line, with the ``# summary:`` and ``# requires:``
  lines left out.

.. seealso::

   - :doc:`quickstart <quickstart>` — installing and the config-driven training
     pattern.
   - :doc:`mlip_guide <mlip_guide>` — what the bundled ``mlip`` example
     actually trains.
   - :mod:`tensorial.reaxkit` — the training, evaluation, and config API the
     command wraps.
