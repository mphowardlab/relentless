"""
===================================
Simulations (`relentless.simulate`)
===================================

`relentless.simulate` implements a generalizable and human-readable
interface for performing molecular simulations. The simulations are used to
evolve a system described by interactions in :mod:`relentless.potential` to
generate statistical ensembles. This module implements code to translate one
common simulation "recipe" into a running simulation in a variety of popular
packages. This gives you the freedom to choose simulation software this is
most compatible with your environment and resources using a single common code!
It also helps document workflows that can be shared and reproduced by others.

.. rubric:: Example
.. code::

    init = relentless.simulate.InitializeRandomly(seed=1)
    avg = relentless.simulate.EnsembleAverage(
            check_thermo_every=5,
            check_rdf_every=5,
            rdf_dr=0.1)
    ops = [relentless.simulate.RunLangevinDynamics(
                steps=1e3,
                timestep=0.001,
                friction=0.8,
                seed=2),
            relentless.simulate.RunLangevinDynamics(
                steps=1e4,
                timestep=0.001,
                friction=0.8,
                seed=3,
                analyzers=avg)
            ]

    # perform simulation using LAMMPS and save ensemble
    lmp = relentless.simulate.LAMMPS(init, ops)
    sim = lmp.run(potentials)
    sim[avg].ensemble.save('ensemble.json')

Engines
=======

.. autosummary::
    :toctree: generated/

    Dilute
    HOOMD
    LAMMPS

Initializers
============

.. autosummary::
    :toctree: generated/

    InitializeFromFile
    InitializeRandomly

Molecular dynamics
==================

.. autosummary::
    :toctree: generated/

    RunMolecularDynamics

Thermostats
-----------

.. autosummary::
    :toctree: generated/

    BerendsenThermostat
    NoseHooverThermostat

Barostats
---------

.. autosummary::
    :toctree: generated/

    BerendsenBarostat
    MTKBarostat

Other dynamics
==============

.. autosummary::
    :toctree: generated/

    MinimizeEnergy
    RunBrownianDynamics
    RunLangevinDynamics

Analyzers
=========

.. autosummary::
    :toctree: generated/

    EnsembleAverage
    Record
    WriteTrajectory

Running a simulation
====================

Results
-------

.. autosummary::
    :toctree: generated/

    SimulationInstance

Defining interactions
---------------------

.. autosummary::
    :toctree: generated/

    Potentials
    PotentialTabulator
    PairPotentialTabulator

Developer classes
=================

.. autosummary::
    :toctree: generated/

    Simulation
    SimulationOperation
    AnalysisOperation
    Barostat
    Thermostat

"""

from .analyze import EnsembleAverage, Record, WriteTrajectory
from .dilute import Dilute
from .hoomd import HOOMD
from .initialize import InitializeFromFile, InitializeRandomly
from .lammps import LAMMPS
from .md import (
    Barostat,
    BerendsenBarostat,
    BerendsenThermostat,
    MinimizeEnergy,
    MTKBarostat,
    NoseHooverThermostat,
    RunBrownianDynamics,
    RunLangevinDynamics,
    RunMolecularDynamics,
    Thermostat,
)
from .simulate import (
    AnalysisOperation,
    PairPotentialTabulator,
    Potentials,
    PotentialTabulator,
    Simulation,
    SimulationInstance,
    SimulationOperation,
)
