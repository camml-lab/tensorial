Quick start
===========

This guide walks you through the canonical tensorial workflow: install, load
the Qm9 dataset, and inspect what a few ``jraph`` graphs look like. The
canonical training path is a Hydra config plus :func:`tensorial.reaxkit.train`;
we cover that next.

.. contents::
   :local:

1. Install
----------

``tensorial`` is a Python package on PyPI. ``reax`` (the training framework)
is installed with it automatically. For the scientific-dataset extras:

.. code-block:: shell

   pip install "tensorial[data]"

For a local dev install with the test toolchain:

.. code-block:: shell

   pip install -e ".[test,data]"

2. Load Qm9 data
----------------

``tensorial.datasets.Qm9`` downloads the official figshare tarball on first
use, then returns either plain molecule dicts or, when ``as_graphs`` is set,
``jraph.GraphsTuple`` graphs built from each molecule:

.. code-block:: python

   from tensorial.datasets import Qm9

   ds = Qm9(
       data_dir="data/",
       limit=500,                                  # subsample for a quick demo
       as_graphs={"r_max": 3.5, "self_edges": False},
   )
   graph = ds[0]
   print(graph.nodes.keys(), "graph:", graph.globals.keys())

If you already have a ``GraphsTuple`` iterable, you can skip this step and
pass your own data straight into a ``reax`` dataloader.

3. Train with a Hydra config
----------------------------

Tensorial training is config-driven. You describe the
datamodule / model / trainer / listeners in a YAML file, then call
:func:`tensorial.reaxkit.train`:

.. code-block:: yaml

   # configs/my_exp.yaml
   data:
     _target_: tensorial.gcnn.data.GraphDataModule
     train:
       _target_: tensorial.gcnn.data.GraphLoader
       dataset: tensorial.datasets.Qm9
   model:
     _target_: my_package.MyQm9Model
   train:
     _target_: reax.stages.Train
     max_epochs: 20

.. code-block:: python

   from tensorial.reaxkit import train
   import hydra

   @hydra.main(version_base=None, config_path="configs", config_name="my_exp")
   def run(cfg):
       return train(cfg)

Tensorial's :func:`tensorial.reaxkit.evaluate` does the same for evaluation
runs.

4. Inspect metrics
------------------

The :mod:`tensorial.reaxkit.listeners` package ships a
:class:`tensorial.reaxkit.listeners.GraphParityPlotter` that writes a
``.pdf`` of true-vs-predicted into the trainer's log directory at the end
of each training stage:

.. code-block:: yaml

   listeners:
     - _target_: tensorial.reaxkit.listeners.GraphParityPlotter
       targets: globals.energy
       fit_plot_every: 5

See the `API reference </api_reference/modules.html>`_ for the full list of
building blocks.
