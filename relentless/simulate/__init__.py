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
    ops = [relentless.simulate.InitializeRandomly(seed=1),
           relentless.simulate.AddLangevinIntegrator(dt=0.1, friction=0.8, seed=2),
           relentless.simulate.Run(steps=1e3),
           relentless.simulate.AddEnsembleAnalyzer(check_thermo_every=5,
                                                   check_rdf_every=5,
                                                   rdf_dr=dr),
           relentless.simulate.RunUpTo(step=1e4)]

    # perform simulation using LAMMPS
    lmp = relentless.simulate.LAMMPS(ops)
    lmp.run(...)

.. rubric:: How it works

To learn more about how to setup a simulation, read through the following:

.. toctree::
    :maxdepth: 1

    engines
    operations
    interactions

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
