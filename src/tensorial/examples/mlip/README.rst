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
