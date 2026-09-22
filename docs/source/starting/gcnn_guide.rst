Building graph networks with the ``tensorial.gcnn`` module
==========================================================

``tensorial.gcnn`` is the biggest, and most-used, part of the library. It
provides equivariant graph message-passing building blocks on top of
``jraph.GraphsTuple``: node- and edge-wise layers, radial and spherical
bases, MACE-style interaction blocks, and graph-aware losses and metrics.

This page is the practical companion to the `API reference
</api_reference/modules.html>`_ and the :doc:`concepts
<concepts>` glossary.

.. contents::
   :local:

Data model
----------

A model operates on a ``jraph.GraphsTuple`` with three attribute maps:

.. code-block:: python

    graph.nodes   # dict[str, Array]   -- per-node attributes
    graph.edges   # dict[str, Array]   -- per-edge attributes (incl. senders/receivers)
    graph.globals # dict[str, Array]   # per-graph attributes

Every key is a *field name*. The layers below read and write fields, so
chaining them is just a matter of feeding one field's output into the next
layer's input. :mod:`tensorial.gcnn.keys` defines a small set of well-known,
conventional field names (e.g. the ``energy`` / ``predicted_energy`` style
pairs) used by the loss and metric helpers, and for PBC structures the
:mod:`tensorial.gcnn.atomic` sub-package defines ``ATOMIC_NUMBERS``,
``ENERGY_PER_ATOM``, ``TOTAL_ENERGY``, ``FORCES``, ``STRESS``, ``VIRIAL`` and
``PBC`` as standard graph keys.

Node-wise layers
----------------

These update one node tensor at a time (no message passing).

NodewiseLinear
--------------

An equivariant linear map applied to a node field. The output type (the
*irreps*) is declared explicitly and tensorial infers the weights for you:

.. code-block:: python

    layer = NodewiseLinear(
        irreps_out="16x0o + 16x0e + 16x1o + 16x1e",
        field="features",          # input field
        out_field="features",      # output field
        num_types=n_elements,      # optional per-type linear maps
    )

NodewiseEncoding
----------------

Turns a set of node attributes (a :class:`tensorial.IrrepsTree` of
independent tensorial modules such as :class:`tensorial.OneHot`) into a single
node tensor under ``out_field``:

.. code-block:: python

    NodewiseEncoding(
        attrs={"species": OneHot(num_classes=n_elements)},
        out_field="node_features",
    )

NodewiseDecoding
----------------

The inverse of :class:`NodewiseEncoding`: decompose a node tensor into a set
of named output tensors (for instance a Born-charge vector and a Raman
tensor) each produced by an independent module:

.. code-block:: python

    NodewiseDecoding(
        in_field="predicted_born_charges",
        attrs={
            "predicted_born_charges": CartesianTensor(
                formula="ij", i="1o", j="1o"
            )
        },
    )

Other helpers: :class:`tensorial.gcnn.NodewiseReduce`,
:class:`tensorial.gcnn.NodewiseEmbedding`.

Edge-wise layers
----------------

These update the per-edge tensors (messages).

EdgeVectors
-----------

Compute the edge vectors (``r_ij``) from node positions and attach them as a
:class:`tensorial.SphericalHarmonic` edge field:

.. code-block:: python

    EdgeVectors()                       # -> edges.edge_vectors : SphericalHarmonic(1o)

EdgewiseEncoding
----------------

Combine a set of per-edge attributes into a single edge tensor, mirroring
:class:`tensorial.gcnn.NodewiseEncoding` on the edge side:

.. code-block:: python

    EdgewiseEncoding(
        attrs={
            "sh": SphericalHarmonic(irreps="...", normalise=True),
        },
        out_field="edge_features",
    )

RadialBasisEdgeEncoding
-----------------------

Expand each edge's length in a learned radial basis (an :class:`RBF`),
optionally gated by an envelope function of radius ``r_max``:

.. code-block:: python

    RadialBasisEdgeEncoding(
        num_basis=8,
        r_max=r_max,
        field="edge_lengths",
        out_field="radial_embeddings",
    )

Equivariant building blocks
---------------------------

The symmetry-typed tensors that feed all of the above layers live in
:mod:`tensorial.tensors`:

- :class:`tensorial.SphericalHarmonic` — an ``l``-order irreps object over a
  rotation group. Created with ``SphericalHarmonic(irreps, normalise,
  normalisation=None)``.
- :class:`tensorial.CartesianTensor` — a Cartesian tensor of any order.
  ``CartesianTensor(formula, **irreps_dict)`` where the formula spells out the
  indices (``"i"``, ``"ij"``, ``"ijk"``, ...) and each index is attached to an
  irrep (``i="1o"``, ``j="1e"``).
- :class:`tensorial.OneHot` — a one-hot vector over ``num_classes`` species.
- :class:`tensorial.NoOp`, :class:`tensorial.AsIrreps` — adapters.

``irreps`` strings use the e3nn-jax convention: a multiplicity, a
dimensionality ``x``, and a parity/pin label (``0e`` scalar-even, ``1o``
vector-odd, etc.). For example ``"16x0e + 16x1o + 16x2e"`` is a 16-d even
scalar channel plus 16-d odd vector plus 16-d even rank-2 tensor.

MACE
----

:class:`tensorial.gcnn.Mace` is a full MACE interaction block: it performs the
node→message→readout update in one layer. It requires an output type, an
output field, and a hidden-irreps spec, plus (for multi-species systems) the
number of species and their mean degree:

.. code-block:: python

    Mace(
        irreps_out="16x0e + 16x1o + 16x1e + 16x2e",
        out_field="node_features",
        hidden_irreps="16x0e + 16x1o + 16x1e + 16x2e",
        num_types=n_elements,
        avg_num_neighbours=avg_num_neighbours,
        # optional:
        correlation_order=3,
        num_interactions=2,
        interaction_irreps="o3_restricted",
    )

For a single layer rather than a full block, use
:class:`tensorial.gcnn.MaceLayer` (or :class:`tensorial.gcnn.InteractionBlock`).
The sibling module :mod:`tensorial.gcnn.nequip` provides
:class:`tensorial.gcnn.NequipLayer` for the NEquIP-style block.

Losses
------

Losses live in :mod:`tensorial.gcnn.losses` and reduce a pair of
``GraphsTuple``s (predictions and targets) to a single scalar. The simplest,
:class:`tensorial.gcnn.Loss`, wraps a pure element-wise loss (any
``optax.losses`` by name, or a custom callable) and applies it to two named
fields of the graph:

.. code-block:: python

    Loss(
        loss_fn=optax.squared_error,
        targets="globals.energy",
        predictions="globals.predicted_energy",
        reduction="mean",
    )

:class:`tensorial.gcnn.WeightedLoss` is a weighted sum of several
:class:`Loss` (or other :class:`GraphLoss`) instances:

.. code-block:: python

    WeightedLoss(
        loss_fns=[
            Loss(loss_fn=optax.squared_error, targets="globals.energy",
                 predictions="globals.predicted_energy"),
            Loss(loss_fn=optax.squared_error, targets="nodes.forces",
                 predictions="nodes.predicted_forces"),
        ],
        weights=[100., 1.],
    )

Metrics
-------

Metrics live in :mod:`tensorial.gcnn.metrics` and are ``reax``-compatible
(``update`` / ``merge`` / ``compute``), so they can be hung off a stage
directly. The most general constructor is :func:`tensorial.gcnn.graph_metric`,
which pairs a ``reax`` metric with the prediction/target fields and an
optional mask. The metric can be given by a `reax` metric name (resolved with
``reax.metrics.get``, e.g. ``rmse``, ``mae``, ``mse``) or a callable:

.. code-block:: python

    graph_metric(
        metric="rmse",
        predictions="globals.predicted_energy",
        targets="globals.energy",
        mask="auto",
    )

The :mod:`tensorial.gcnn.atomic` sub-package provides PBC-structure
specialisations: :class:`EnergyContributionLstsq`,
:class:`EnergyPerAtomLstsq`, :class:`AvgNumNeighboursByAtomType`,
:class:`ForceStd`, and dataset statistics (``AllAtomicNumbers``,
``NumSpecies``, ``AvgNumNeighbours``).

Wiring a model
--------------

The layers above are all ``flax.nnx`` / ``equinox`` modules. The simplest
way to chain them is :class:`tensorial.nn.Sequential`, which preserves
``GraphsTuple`` structure across the layers:

.. code-block:: python

    import tensorial as t
    import tensorial.nn as nn

    model = nn.Sequential([
        atomic.SpeciesTransform(atomic_numbers=Z),
        t.gcnn.NodewiseEncoding(attrs={"species": t.OneHot(num_classes=n_species)}),
        t.gcnn.EdgeVectors(),
        t.gcnn.EdgewiseEncoding(attrs={"sh": t.SphericalHarmonic(irreps="1o", normalise=True)}),
        t.gcnn.Mace(irreps_out=out_ir, out_field="node_features",
                    hidden_irreps=hidden_ir, num_types=n_species),
        t.gcnn.NodewiseLinear(irreps_out="1x0e + 1x1e", out_field="predicted_born"),
    ])

In a training config you would normally build the model with a ``_target_``
key so it composes with ``Hydra`` (see the
:doc:`quickstart <quickstart>` for the standard pattern).

End-to-end: a MACE model in a Hydra config
-------------------------------------------

Below is a complete, self-contained configuration for training a
:class:`tensorial.gcnn.Mace` block to predict energy, forces, Born charges,
and Raman tensors. It composes a :class:`tensorial.reaxkit.ReaxModule`, a
data-driven model block, a weighted loss, per-field metrics, and an
``optax`` optimizer.

``${from_data.X}`` interpolations are filled by the
:class:`tensorial.reaxkit.from_data.FromData` stage at the start of training:
each name is a ``reax``-compatible metric evaluated on the training
dataloader, and the result is baked into the config before the rest of the
pipeline is instantiated.

.. code-block:: yaml

    # train.py:  python train.py --config-path configs/my_exp.yaml

    model:
      _target_: tensorial.reaxkit.ReaxModule
      jit: True
      donate_graph: True

      # --- dataset statistics, baked in by the `from_data` stage ------------
      # each entry is either a `_target_` (instantiated via Hydra) or a bare
      # reax metric name resolved with `reax.metrics.get`
      from_data:
        # which species are in the dataset (e.g. [1, 6, 7, 8])
        atomic_numbers:
          _target_: tensorial.gcnn.atomic.AllAtomicNumbers
        # number of distinct species
        n_elements:
          _target_: tensorial.gcnn.atomic.NumSpecies
        # mean number of neighbours (used by the MACE readout normalisation)
        avg_num_neighbours:
          _target_: tensorial.gcnn.atomic.AvgNumNeighbours

      data:
        train:
          _target_: tensorial.gcnn.data.GraphLoader
          datamodule: ...
        validation:
          _target_: tensorial.gcnn.data.GraphLoader
          datamodule: ...
        test:
          _target_: tensorial.gcnn.data.GraphLoader
          datamodule: ...

      # ---------------------------------------------------------------------
      metrics:
        energy:
          _target_: tensorial.gcnn.graph_metric
          metric: rmse
          targets: globals.energy
          predictions: globals.predicted_energy
        raman:
          _target_: tensorial.gcnn.graph_metric
          metric: rmse
          targets: nodes.raman
          predictions: nodes.predicted_raman

      loss_fn:
        _target_: tensorial.gcnn.WeightedLoss
        loss_fns:
          - _target_: tensorial.gcnn.Loss
            loss_fn: squared_error
            targets: globals.energy
            predictions: globals.predicted_energy
          - _target_: tensorial.gcnn.Loss
            loss_fn: squared_error
            targets: nodes.forces
            predictions: nodes.predicted_forces
        weights: [100., 1.]

      optimizer:
        _target_: optax.adam
        learning_rate: 1.0e-3

      # Optional scheduler (uncomment to use):
      # scheduler:
      #   _target_: optax.schedules.cosine_decay_schedule
      #   decay_steps: 100_000
      #   alpha: 0.0

      model:
        _target_: tensorial.nn.Sequential
        layers:
          # 1. map Z -> one-hot species vector
          - _target_: tensorial.gcnn.atomic.SpeciesTransform
            atomic_numbers: ${from_data.atomic_numbers}

          # 2. node embedding layer
          - _target_: tensorial.gcnn.NodewiseEncoding
            attrs:
              species:
                _target_: tensorial.OneHot
                num_classes: ${from_data.n_elements}

          # 3. edge vectors from positions
          - _target_: tensorial.gcnn.EdgeVectors

          # 4. encode the SH (1o) edge field, normalised
          - _target_: tensorial.gcnn.EdgewiseEncoding
            attrs:
              sh:
                _target_: tensorial.SphericalHarmonic
                irreps: "1o"
                normalise: True

          # 5. radial basis of the edge length
          - _target_: tensorial.gcnn.RadialBasisEdgeEncoding
            num_basis: 16
            r_max: 5.0
            out_field: radial_embeddings

          # 6. first node linear projection to the MACE input space
          - _target_: tensorial.gcnn.NodewiseLinear
            field: node_features
            irreps_out: "16x0o + 16x0e + 16x1o + 16x1e"
            num_types: ${from_data.n_elements}

          # 7. the MACE block itself
          - _target_: tensorial.gcnn.Mace
            irreps_out: "16x0o + 16x0e + 16x1o + 16x1e"
            out_field: node_features
            interaction_irreps: o3_restricted
            hidden_irreps: "16x0o + 16x1o + 16x1e + 16x3e"
            num_types: ${from_data.n_elements}
            avg_num_neighbours: ${from_data.avg_num_neighbours}

          # 8. Born-charge head (scalar + vector)
          - _target_: tensorial.gcnn.NodewiseLinear
            field: node_features
            irreps_out: "4x0e + 2x1o"
            out_field: predicted_born_charges

          - _target_: tensorial.gcnn.NodewiseDecoding
            in_field: predicted_born_charges
            attrs:
              predicted_born_charges:
                _target_: tensorial.CartesianTensor
                formula: ij
                i: 1o
                j: 1o

          # 9. Raman-tensor head (rank-2 symmetric)
          - _target_: tensorial.gcnn.NodewiseLinear
            field: node_features
            irreps_out: "2x1o + 1x2o + 1x3o"
            out_field: predicted_raman

          - _target_: tensorial.gcnn.NodewiseDecoding
            in_field: predicted_raman
            attrs:
              predicted_raman:
                _target_: tensorial.CartesianTensor
                formula: ijk
                i: 1o
                j: 1o
                k: 1o

.. note::
   The example above is the shape of the intended end-to-end configuration.
   A few ``${from_data.X}`` keys (for instance ``mul`` and ``ell_max`` used
   to construct :class:`tensorial.SphericalHarmonic` irreps) would normally
   come from a small :mod:`tensorial.irreps_builder` helper that is not
   yet part of the public API. Until that lands, either fix the values in
   your config, or compute them once and interpolate them into the
   ``attrs`` dictionaries by hand.

.. seealso::

   - :mod:`tensorial.gcnn` — the full sub-package, including the sub-modules
     :mod:`tensorial.gcnn.atomic`, :mod:`tensorial.gcnn.data`,
     :mod:`tensorial.gcnn.losses`, :mod:`tensorial.gcnn.metrics`,
     :mod:`tensorial.gcnn.mace`, :mod:`tensorial.gcnn.nequip`, and
     :mod:`tensorial.gcnn.experimental`.
   - :doc:`concepts <concepts>` — the vocabulary (``GraphsTuple``,
     irreps, radial/spherical bases, message passing).
   - :doc:`quickstart <quickstart>` — the minimal training loop.
