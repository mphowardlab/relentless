"""
Simulations
===========

.. toctree::
    :maxdepth: 1

    dilute
    hoomd
    lammps

A generalizable and human-readable interface for molecular simulations is provided.
Molecular simulations are used to evolve a system described by a potential and
generate ensemble-averaged descriptions of the system's properties. These act as
figures of merit which are used to minimize the statistical distance between ensembles
during the inverse design process. In addition, a framework for generic simulation
operations is provided to ensure that users have the freedom to choose between
simulation engines without significantly altering their code.

Example
-------
Implement a sequence of simulation operations through a reproducible workflow that
can be directly translated between all of the supported simulation packages::

    # generic simulation operations
    ops = [relentless.simulate.InitializeRandomly(seed=1),
           relentless.simulate.AddLangevinIntegrator(dt=0.1, friction=0.8, seed=2),
           relentless.simulate.Run(steps=1e3),
           relentless.simulate.AddEnsembleAnalyzer(check_thermo_every=5,
                                                   check_rdf_every=5,
                                                   rdf_dr=dr),
           relentless.simulate.RunUpTo(step=1e4)]

    # perform operations using HOOMD and LAMMPS
    hmd = relentless.simulate.HOOMD(ops)
    lmp = relentless.simulate.LAMMPS(ops)

Molecular simulation runs are performed in a :class:`Simulation` ensemble container,
which initializes and runs a set of :class:`SimulationOperation`\s. Each simulation
run requires the input of an ensemble, the interaction potentials, and a directory
to write the output data, which are all used to construct a :class:`SimulationInstance`.

The simulations can use a combination of multiple :class:`~relentless.potential.potential.Potential`\s or
:class:`~relentless.potential.pair.PairPotential`\s tabulated together, the interface for
which is given here using :class:`Potentials` and the tabulators :class:`PotentialTabulator`
and :class:`PairPotentialTabulator`.

A number of :class:`Thermostat`\s and :class:`Barostat`\s are also provided to enable
control of the temperature and pressure of the simulation.

.. autosummary::
    :nosignatures:

    Simulation
    SimulationInstance
    Potentials
    PotentialTabulator
    PairPotentialTabulator
    BerendsenThermostat
    NoseHooverThermostat
    BerendsenBarostat
    MTKBarostat

The following generic simulation operations have been implemented:

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

.. rubric:: Developer notes

To implement your own simulation operation, create a class that derives from
:class:`SimulationOperation` and define the required methods.

To implement your own thermostat or barostat, create a class that derives from
:class:`Thermostat` or :class:`Barostat` and define the required methods.

.. autosummary::
    :nosignatures:

    SimulationOperation
    Thermostat
    Barostat

.. autoclass:: Simulation
    :members:
.. autoclass:: SimulationInstance
    :members:
.. autoclass:: SimulationOperation
    :members:
.. autoclass:: Potentials
    :members:
.. autoclass:: PotentialTabulator
    :members:
.. autoclass:: PairPotentialTabulator
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

"""
from .simulate import (Simulation,
                       SimulationInstance,
                       SimulationOperation,
                       Potentials,
                       PotentialTabulator,
                       PairPotentialTabulator,
                       Thermostat,
                       BerendsenThermostat,
                       NoseHooverThermostat,
                       Barostat,
                       BerendsenBarostat,
                       MTKBarostat)
from .simulate import (InitializeFromFile,
                      InitializeRandomly,
                      MinimizeEnergy,
                      AddBrownianIntegrator,
                      RemoveBrownianIntegrator,
                      AddLangevinIntegrator,
                      RemoveLangevinIntegrator,
                      AddVerletIntegrator,
                      RemoveVerletIntegrator,
                      Run,
                      RunUpTo,
                      AddEnsembleAnalyzer,
                      )

from . import dilute
from .dilute import Dilute

from . import hoomd
from .hoomd import HOOMD

from . import lammps
from .lammps import LAMMPS
