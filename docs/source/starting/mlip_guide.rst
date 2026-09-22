Building machine-learning interatomic potentials with ``tensorial.mlip``
========================================================================

``tensorial.mlip`` is a thin, higher-level layer on top of
:mod:`tensorial.gcnn` for the specific task of **machine-learning
interatomic potentials** (MLIPs): predict the *total energy* of a structure
and, by automatic differentiation, the *forces* and *stresses* as well.

The design is deliberately small. You build an energy model with the
:mod:`tensorial.gcnn` building blocks (see the :doc:`gcnn_guide
<gcnn_guide>`), then wrap it with ``mlip`` blocks to get the derived
quantities and the standard MLIP loss and metrics:

.. contents::
    :local:

Derived quantities
------------------

The physics of a potential is that

.. math::
    :nowrap:

    \mathbf{F} = -\frac{\partial E}{\partial \mathbf{r}}
    \qquad\qquad
    \sigma = -\frac{1}{V} \frac{\partial E}{\partial \mathbf{h}}

so forces and stresses fall out of the energy for free. ``mlip`` exposes two
``flax.linen`` blocks that do exactly that:

- :class:`tensorial.mlip.CalcForces` — differentiates the energy with respect
  to node positions and writes the result to a ``nodes.<out_key>`` field.
- :class:`tensorial.mlip.CalcStresses` — differentiates the energy with
  respect to the cell (volume-normalised) and writes the per-graph stress to a
  ``globals.<out_key>`` field.

Both take an ``energy_fn`` (any ``gcnn.GraphFunction`` that populates the
energy field) and an ``out_key``. The default output keys come from
:mod:`tensorial.mlip.keys`, which re-exports the canonical
:mod:`tensorial.gcnn.atomic`/``gcnn`` keys (``FORCES``, ``STRESS``,
``TOTAL_ENERGY``, ``VIRIAL``, ...).

Wiring the blocks
-----------------

Chain the model, the force block, and the stress block with
:class:`tensorial.nn.Sequential` (which preserves ``GraphsTuple`` structure):

.. code-block:: python

    import tensorial as t
    import tensorial.nn as nn
    from tensorial.mlip import CalcForces, CalcStresses

    energy_model = nn.Sequential([...your gcnn energy layers...])
    model = nn.Sequential([
        energy_model,
        CalcForces(energy_fn=energy_model),          # adds nodes.predicted_forces
        CalcStresses(energy_fn=energy_model),        # adds globals.predicted_stress
    ])

For energy-only (forces/stresses omitted, or PBC-free) systems you can use a
plain ``gcnn`` model; ``mlip`` is what gives you the derivatives.

Loss
----

:func:`tensorial.mlip.mlip_loss` builds a :class:`tensorial.gcnn.WeightedLoss`
over the energy and force terms, with the conventional relative weighting
(energy ``1.0``, forces ``20.0``) out of the box:

.. code-block:: python

    from tensorial.mlip import mlip_loss

    loss = mlip_loss(energy=True, forces=True, energy_weight=1.0, force_weight=20.0)

Any term can be switched off (``energy=False`` / ``forces=False``) if you only
train on a subset.

Metrics
-------

:mod:`tensorial.mlip.metrics` ships four ready-made, ``reax``-compatible
graph metrics, normalised by particle count where appropriate:

- :class:`tensorial.mlip.EnergyPerAtomRmse`
- :class:`tensorial.mlip.EnergyPerAtomMae`
- :class:`tensorial.mlip.ForceRmse`
- :class:`tensorial.mlip.StressRmse`

These are the canonical MLIP quality numbers (eV/atom for energy, eV/Å for
forces) and can be hung off a training stage directly.

Data
----

:mod:`tensorial.mlip.data` provides two ``reax`` datamodules:

- :class:`tensorial.mlip.AtomGraphModule` — wrap an existing iterable of
  ``jraph.GraphsTuple`` into train/val/test loaders with automatic padding.
- :class:`tensorial.mlip.Qm9DataModule` — download and parse the QM9
  (134k-molecule) dataset, converting each molecule to a graph for you.

A simple potential
------------------

If you don't have a trained model yet,
:class:`tensorial.mlip.LennardJonesEnergy` is a tiny, analytic Lennard-Jones
energy layer — useful as a sanity check or a baseline.

ASE calculator
--------------

:mod:`tensorial.mlip.calculators.ase` exposes an ``ase.calculators.Calculator``
so a trained ``mlip`` model can be dropped straight into an ``ASE``
workflow (``atoms.get_potential_energy()``, ``atoms.get_forces()``, ...):

.. code-block:: python

    from ase import Atoms
    from tensorial.mlip.calculators import ase

    calc = ase.Calculator.from_checkpoint(
        config_path="configs/my_exp.yaml",
        checkpoint_path="checkpoints/...",
        r_max=5.0,
    )
    atoms = Atoms("SiSi", positions=[[0, 0, 0], [2.35, 0, 0]], cell=(6, 6, 6), pbc=True)
    atoms.calc = calc
    print(atoms.get_potential_energy(), atoms.get_forces())

I/O helpers
-----------

:mod:`tensorial.mlip.io.n2p2` contains ``write_n2p2`` / ``read_n2p2`` for the
N2P2 ``.data`` text format (positions, forces, cell).

Further reading
---------------

See the ``tensorial.mlip`` package in the
`API reference </api_reference/modules.html>`_ for the full list of members,
and the :doc:`gcnn_guide <gcnn_guide>` for the ``gcnn`` blocks that build the
energy model itself.
