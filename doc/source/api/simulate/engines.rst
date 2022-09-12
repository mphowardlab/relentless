.. py:currentmodule:: relentless.simulate

Engines
-------
Molecular simulation runs are performed by a :class:`Simulation` engine.
The following simulation backends have been implemented:

.. autosummary::
    :nosignatures:

    Dilute
    HOOMD
    LAMMPS

.. autoclass:: Simulation
    :members:
.. autoclass:: SimulationInstance
    :members:
.. autoclass:: SimulationOperation
    :members:
.. autoclass:: Dilute
    :members:
.. autoclass:: HOOMD
    :members:
.. autoclass:: LAMMPS
    :members:
    :exclude-members: to_commands
