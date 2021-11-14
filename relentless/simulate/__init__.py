"""
Simulations
===========

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

.. automodule:: relentless.simulate.simulate

.. automodule:: relentless.simulate.generic

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

from . import dilute
from .dilute import Dilute

from . import hoomd
from .hoomd import HOOMD

from . import lammps
from .lammps import LAMMPS

# setup generic (adapter) objects
from . import generic
from .generic import (InitializeFromFile,
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
generic.GenericOperation.add_backend(Dilute)
generic.GenericOperation.add_backend(HOOMD)
generic.GenericOperation.add_backend(LAMMPS)
