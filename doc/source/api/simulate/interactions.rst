.. py:currentmodule:: relentless.simulate

Interactions
------------
To :meth:`~Simulation.run` a simulation, you need to specify the interactions.
This allows the same sequence of operations to produce different output,
depending on how the interactions are parametrized. The interactions
are collected together in the :class:`Potentials`, using specialized tabulators
that help with the translation process.

.. autosummary::
    :nosignatures:

    Potentials
    PotentialTabulator
    PairPotentialTabulator

A :meth:`Simulation.run` will produce a :class:`SimulationInstance`, which
contains a specific instance of simulation data.

.. autoclass:: Potentials
    :members:
.. autoclass:: PotentialTabulator
    :members:
.. autoclass:: PairPotentialTabulator
    :members:
