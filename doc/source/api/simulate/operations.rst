.. py:currentmodule:: relentless.simulate

Operations
----------
To execute the simulation, you need to specify a sequence of operations. These
operations will be translated by the engine. Each :class:`Simulation` *may*
(but is not required to!) implement the following:

.. autosummary::
    :nosignatures:

    InitializeFromFile
    InitializeRandomly
    MinimizeEnergy
    RunBrownianDynamics
    RunLangevinDynamics
    RunMolecularDynamics
    EnsembleAverage

.. autoclass:: InitializeFromFile
    :members:
.. autoclass:: InitializeRandomly
    :members:
.. autoclass:: MinimizeEnergy
    :members:
.. autoclass:: RunBrownianDynamics
    :members:
.. autoclass:: RunLangevinDynamics
    :members:
.. autoclass:: RunMolecularDynamics
    :members:
.. autoclass:: EnsembleAverage
    :members:
.. autoclass:: Thermostat
    :members:
.. autoclass:: BerendsenThermostat
    :members:
.. autoclass:: NoseHooverThermostat
    :members:
.. autoclass:: Barostat
    :members:
.. autoclass:: BerendsenBarostat
    :members:
.. autoclass:: MTKBarostat
    :members:
