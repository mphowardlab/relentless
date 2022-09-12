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
    AddBrownianIntegrator
    RemoveBrownianIntegrator
    AddLangevinIntegrator
    RemoveLangevinIntegrator
    AddVerletIntegrator
    RemoveVerletIntegrator
    Run
    RunUpTo
    AddEnsembleAnalyzer

.. autoclass:: InitializeFromFile
    :members:
.. autoclass:: InitializeRandomly
    :members:
.. autoclass:: MinimizeEnergy
    :members:
.. autoclass:: AddBrownianIntegrator
    :members:
.. autoclass:: RemoveBrownianIntegrator
    :members:
.. autoclass:: AddLangevinIntegrator
    :members:
.. autoclass:: RemoveLangevinIntegrator
    :members:
.. autoclass:: AddVerletIntegrator
    :members:
.. autoclass:: RemoveVerletIntegrator
    :members:
.. autoclass:: Run
    :members:
.. autoclass:: RunUpTo
    :members:
.. autoclass:: AddEnsembleAnalyzer
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
