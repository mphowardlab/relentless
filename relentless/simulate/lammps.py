import abc
import os
from packaging import version

import numpy as np

from relentless.core.volume import TriclinicBox
from . import simulate

try:
    import lammps
    _lammps_found = True
except ImportError:
    _lammps_found = False

class LAMMPS(simulate.Simulation):
    def __init__(self, operations, **options):
        if not _lammps_found:
            raise ImportError('LAMMPS not found.')

    def _new_instance(self, ensemble, potentials, directory):
        sim = super()._new_instance(ensemble,potentials,directory)

        # create lammps instance with all output disabled
        quiet_launch = ['-echo','none',
                        '-log','none',
                        '-screen','none',
                        '-nocite']
        sim.lammps = lammps.lammps(cmdargs=quiet_launch)

        # lammps uses 1-indexed ints for types, so build mapping in both direction
        sim.type_map = {}
        sim.typeid_map = {}
        for i,t in enumerate(sim.ensemble.types):
            sim.type_map[t] = i+1
            sim.typeid_map[sim.type_map[t]] = t

        return sim

class LAMMPSOperation(simulate.SimulationOperation):
    def __call__(self, sim):
        cmds = self.to_commands(sim)
        sim.lammps.commands_list(cmds)

    @abc.abstractmethod
    def to_commands(self, sim):
        pass

class Initialize(LAMMPSOperation):
    def __init__(self, neighbor_buffer):
        self.neighbor_buffer = neighbor_buffer

    def extract_box_params(self, sim):
        # cast simulation box in LAMMPS parameters
        V = sim.ensemble.V
        if V is None:
            raise ValueError('Box volume must be set.')
        elif not isinstance(V, TriclinicBox):
            raise TypeError('HOOMD boxes must be derived from TriclinicBox')

        Lx = V.a[0]
        Ly = V.b[1]
        Lz = V.c[2]
        xy = V.b[0]
        xz = V.c[0]
        yz = V.c[1]

        lo = -0.5*np.array([Lx,Ly,Lz])
        hi = lo + V.a + V.b + V.c
        return np.concatenate((lo,hi,(xy,xz,yz)))

    def attach_potentials(self, sim):
        cmds = ["neighbor {} multi".format(self.neighbor_width)]
        cmds += ['pair_style table linear {}'.format(len(sim.potentials.pair.r))]
        for i.j in sim.ensemble.pairs:
            # these will need to be written into (potentially temporary) files and read in
            # make sure to drop any entries with r = 0, since these are not allowed in lammps
            pass
        return cmds

class InitializeRandomly(Initialize):
    def __init__(self, neighbor_buffer, seed, units="lj", atom_style="atomic"):
        super().__init__(neighbor_buffer, units, atom_style)
        self.seed = seed
        self.units = units
        self.atom_style = atom_style

    def to_commands(self, sim):
        cmds = ["units {}".format(self.units),
                "boundary p p p",
                "atom_style {}".format(self.atom_style)]

        # make box from ensemble
        box = self.extract_box_params(sim)
        if not np.all(np.isclose(box[-3:],0)):
            cmds += ["region box prism {} {} {} {} {} {} {} {} {}".format(*box)]
        else:
            cmds += ["region box block {} {} {} {} {} {}".format(*box[:-3])]
        cmds += ["create_box {} box".format(len(sim.ensemble.types))]

        # use lammps random initialization routines
        for i in sim.ensemble.types:
            cmds += ["create_atoms {typeid} random {N} {seed} box".format(typeid=sim.type_map[i],
                                                                          N=sim.ensemble.N[i],
                                                                          seed=self.seed+sim.sim.type_map[i])]
        cmds += ["mass * 1.0",
                 "velocity all create {} {}".format(sim.ensemble.T,self.seed)]

        cmds += self.attach_potentials(sim)

        return cmds

