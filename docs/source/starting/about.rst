About tensorial
===============

``tensorial`` is a JAX-based machine learning library for data with geometric,
symmetry, or tensor structure. It is designed to work on top of
`JAX <https://github.com/google/jax>`_, `flax.linen <https://flax.readthedocs.io>`_,
and the `reax <https://camml-lab.github.io/reax>`_ training framework.

Why ``tensorial``?
------------------

Many science and engineering datasets are not tabular or image-like. They live
on graphs (molecules, materials, reaction networks) or carry geometric
quantities (vectors, rotation matrices, second- and higher-order tensors) whose
meaning depends explicitly on a reference frame. A standard ``nn.Linear`` layer
can learn to read those objects, but it does so in a way that is not invariant
to the symmetries that physics expects the system to respect: rotating the
inputs should rotate the outputs, and the score of an invariant model should
not change with the coordinate choice.

Tensorial layers and utilities are built to make this explicit. They:

- track the **symmetry type** of every tensor in the graph (irreducible
  representation of the rotation group, or plain scalars),
- implement **equivariant** operations (linear maps, radial bases, products of
  tensor components) whose output type is inferred from the inputs,
- expose a **graph message-passing** interface (compatible with
  ``jraph.GraphsTuple``) with per-node and per-edge modules, and
- integrate with `reax <https://camml-lab.github.io/reax>`_ for a familiar
  ``fit`` / ``validate`` / ``test`` training loop on multi-device JAX.

What is inside
--------------

- :mod:`tensorial.tensors` - typed tensor containers (``SphericalHarmonic``,
  ``CartesianTensor``, ``OneHot``) with arithmetic and products.
- :mod:`tensorial.gcnn` - graph neural network building blocks: message
  passing, radial and spherical bases, losses, and derivable (differentiable)
  graph functions.
- :mod:`tensorial.signals` - spherical-harmonic and radial
  basis-expansion machinery.
- :mod:`tensorial.geometry` - periodic boundary handling and neighbour
  finding for lattice structures.
- :mod:`tensorial.datasets` - ready-to-use scientific datasets on top of
  ``jraph`` (Qm9, ...).
- :mod:`tensorial.reaxkit` - a ``ReaxModule`` subclass specialised for
  tensorial training tasks: config-driven training, hyper-logging, metrics
  listeners, and parity plotting.
- :mod:`tensorial.nn` / :mod:`tensorial.nn_utils` - helpers for wiring JAX /
  flax.nnx modules into reax stages.

Who is it for?
--------------

Anyone who needs to train deep models on molecular, geometric, or
tensor-valued data in JAX. It is a library, not a full-featured trainer, so
it composes cleanly with any existing JAX / flax pipeline.
