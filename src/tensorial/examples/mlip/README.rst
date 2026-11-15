mlip: E(3)-equivariant interatomic potential example
=====================================================

An E(3)-equivariant machine-learning interatomic potential, using a MACE
(default) or NequIP model, trained on energies (and, where the dataset has
them, forces and stresses).

Data
----
The dataset is chosen by the ``data`` config group, and this example comes in
one variant per dataset in ``data/``. Run ``tensorial examples mlip`` to list
them, or ``tensorial examples mlip.<variant>`` for the full description of one:

- ``si`` (the default): a short trajectory of bulk silicon, bundled as
  ``data/sitraj.xyz`` so it runs offline. Energies, forces and stresses.
- ``qm9``: QM9 small organic molecules (a 10,000-molecule subset by default),
  trained on the energy U0.
- ``mad``: the Massive Atomic Diversity dataset (crystals, surfaces, molecules).
- ``rmd17``: revised MD17 small molecules; pick one with ``data.molecule=``
  and one of the five standard splits with ``data.split=``.
- ``3bpa``: 3BPA, trained at 300 K; override ``data.test=`` to test at 600 K
  or 1200 K.

All but ``si`` are downloaded on first use; ``mad``, ``rmd17`` and ``3bpa``
use the published train/test splits.

``tensorial examples init mlip.<variant>`` sets that variant as the default in
``train.yaml`` and ``eval.yaml``. To switch dataset afterwards, pass ``data=<variant>``
on the command line (its ``data/<variant>.yaml`` must be present). To train on
your own trajectory instead, point the ``si`` configuration at it::

    data=si data.path=/path/to/trajectory.extxyz

Run
---
To train::

    tensorial train -i train.yaml

To evaluate a trained model on the test set, point ``predict`` at the training run's
directory, its checkpoint and its saved config::

    tensorial predict -i logs/train/<run> ckpt_path=logs/train/<run>/checkpoints/last.ckpt \
        +config_path=logs/train/<run>/config.yaml test=true

For a quick check that everything runs (one epoch on a few structures, then testing)::

    tensorial train -i train.yaml debug=smoke

You can select an alternative model or override any option on the
command line, e.g. ``tensorial train -i train.yaml model=nequip r_max=4.0``.
