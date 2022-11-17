"""
Simulations
===========

:mod:`relentless.simulate` implements a generalizable and human-readable
interface for performing molecular simulations. The simulations are used to
evolve a system described by interactions in :mod:`relentless.potential` to
generate statistical ensembles. This module implements code to translate one
common simulation "recipe" into a running simulation in a variety of popular
packages. This gives you the freedom to choose simulation software this is
most compatible with your environment and resources using a single common code!
It also helps document workflows that can be shared and reproduced by others.

.. rubric:: Example
.. code::

    # generic simulation operations
    avg = relentless.simulate.EnsembleAverage(
            check_thermo_every=5,
            check_rdf_every=5,
            rdf_dr=0.1)
    ops = [relentless.simulate.InitializeRandomly(seed=1),
           relentless.simulate.RunLangevinDynamics(
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
    lmp = relentless.simulate.LAMMPS(ops)
    sim = lmp.run(potentials)
    sim[avg].ensemble.save('ensemble.json')

.. rubric:: How it works

To learn more about how to setup a simulation, read through the following:

.. toctree::
    :maxdepth: 1

    engines
    operations
    interactions

"""
from .simulate import (
    AnalysisOperation,
    PairPotentialTabulator,
    PotentialTabulator,
    Potentials,
    Simulation,
    SimulationInstance,
    SimulationOperation,
    )

from .analyze import (
    EnsembleAverage,
    )

from .initialize import (
    InitializeFromFile,
    InitializeRandomly,
    )

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

from .dilute import Dilute

from .hoomd import HOOMD

from .lammps import LAMMPS
