Concepts
========

This page explains the terms we use throughout the API reference. It is a
glossary, not a tutorial.

.. contents::
   :local:

Graphs (``jraph.GraphsTuple``)
------------------------------

Tensorial operates on ``jraph.GraphsTuple`` records as produced by the
``data`` sub-packages. A minimal graph looks like:

.. code-block:: python

   class GraphsTuple(NamedTuple):
       nodes: Dict[str, Any]   # (num_nodes, ...) attributes
       edges: Dict[str, Any]   # (num_edges,) attributes (senders, receivers)
       globals: Dict[str, Any] # scalar / graph-level attributes

Each key in ``nodes`` / ``edges`` / ``globals`` points at a JAX array; the
keys are user-defined (for example ``species``, ``positions``, ``energy``) and
are used by the rest of the library to select which tensor to read or write.
The :mod:`tensorial.gcnn.keys` module defines a small set of well-known key
pairs (``energy`` / ``predicted_energy`` style) used by the loss and metric
helpers.

Tensors and symmetry types
--------------------------

A tensor has a *symmetry type*, i.e. the irreducible representation of the
rotation group that the components transform under. Tensorial currently
supports:

* **Scalars** (``l = 0``) — invariant to rotation.
* **Vectors** (``l = 1``) — transform under a rotation matrix.
* **Spherical harmonics** — general ``l``-order irreps, implemented with
  the :class:`tensorial.tensors.SphericalHarmonic` container.

The :class:`tensorial.tensors.CartesianTensor` class is a convenience wrapper
around a regular ``jax.Array`` whose rows are the Cartesian components of a
tensor of some order; it carries the same metadata and is the type most users
work with.

Bases: radial and spherical
---------------------------

To keep the parameter count manageable, tensorial expresses equivariant
layers in terms of **radial** dependence (a function of the bond length)
times **spherical** dependence (the angular part that determines the tensor
type).

* :class:`tensorial.signals.radials.RadialBasis` — a parameterised
  ``f(r) -> R^k`` radial function.
* :class:`tensorial.signals.bases.SphericalBasis` — the angular expansion
  (a set of irreps).

Combining the two with an :mod:`tensorial.signals.functions` operator gives a
full equivariant kernel.

Message passing
---------------

``tensorial.gcnn`` provides :class:`MessagePassingConvolution` (in
:mod:`tensorial.gcnn._message_passing`) as the standard inter-node update:
an edge-wise linear read-out followed by a node-wise aggregation. The
:class:`tensorial.gcnn._modules.Rescale` and
``IndexedRescale`` helpers handle the bookkeeping of normalising a node
tensor before it is read by an equivariant layer.

ReaxModule
----------

``ReaxModule`` is a subclass of ``reax.Module`` that adds:

* parameter initialisation helpers (``_configure_model``),
* ``prepare_batch`` / ``log`` overrides for the tensorial data shapes,
* and a small collection of listeners (parity plotter, metrics printer).

You subclass it to define ``training_step``, ``validation_step``,
``test_step`` and ``predict_step``.

Config-driven training
----------------------

Everything in :mod:`tensorial.reaxkit` is driven by an
``omegaconf.DictConfig`` that is typically composed by Hydra. Three entry
points are exposed:

* :func:`tensorial.reaxkit.train` — fit + test + predict in one call.
* :func:`tensorial.reaxkit.evaluate` — evaluation-only (no training).
* :func:`tensorial.reaxkit.cli` — a thin CLI wrapper around both.

The optional :class:`tensorial.reaxkit.from_data.FromData` stage lets you
compute statistics on the training data at runtime and bake them back into
the config (for example normalising a target before it is used downstream).

Losses and metrics
------------------

* Losses live in :mod:`tensorial.gcnn.losses` and are ``jax``-compatible
  callable objects.
* Metrics live in :mod:`tensorial.gcnn.metrics` and are ``reax``-compatible
  (they can be aggregated over a stage with ``merge`` / ``update`` /
  ``compute``).
