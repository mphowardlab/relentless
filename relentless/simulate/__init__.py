"""
Simulations
===========

A generalizable and human-readable interface for molecular simulations is provided.
Molecular simulations are used to evolve a system described by a potential and
generate ensemble-averaged descriptions of the system's properties. A framework
for generic simulation operations is provided:

.. toctree::
    :maxdepth: 1

    generic

These operations are compatible with the following simulation interfaces:

.. toctree::
    :maxdepth: 1

    dilute
    hoomd
    lammps

.. automodule:: relentless.simulate.simulate

"""
from .simulate import (Simulation,
                       SimulationInstance,
                       SimulationOperation,
                       Potentials,
                       PotentialTabulator,
                       PairPotentialTabulator)

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
                      AddNPTIntegrator,
                      RemoveNPTIntegrator,
                      AddNVTIntegrator,
                      RemoveNVTIntegrator,
                      Run,
                      RunUpTo,
                      AddEnsembleAnalyzer,
                      )
generic.GenericOperation.add_backend(Dilute)
generic.GenericOperation.add_backend(HOOMD)
generic.GenericOperation.add_backend(LAMMPS)
