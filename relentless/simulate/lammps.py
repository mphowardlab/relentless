import abc
import os

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

        super().__init__(operations,**options)

    def _new_instance(self, ensemble, potentials, directory):
        sim = super()._new_instance(ensemble,potentials,directory)

        # create lammps instance with all output disabled
        #quiet_launch = ['-echo','none',
        #                '-log','none',
        #                '-screen','none',
        #                '-nocite']
        sim.lammps = lammps.lammps()#cmdargs=quiet_launch)

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
            raise TypeError('LAMMPS boxes must be derived from TriclinicBox')

        Lx = V.a[0]
        Ly = V.b[1]
        Lz = V.c[2]
        xy = V.b[0]
        xz = V.c[0]
        yz = V.c[1]

        lo = -0.5*np.array([Lx,Ly,Lz])
        hi = lo + V.a + V.b + V.c

        return np.array([lo[0],hi[0],lo[1],hi[1],lo[2],hi[2],xy,xz,yz])

    def attach_potentials(self, sim):
        #tabulate, drop entries where r = 0
        k = np.where(sim.potentials.pair.r==0)
        r = np.delete(sim.potentials.pair.r,k)

        cmds = ['neighbor {skin} multi'.format(skin=self.neighbor_buffer)]
        cmds += ['pair_style table linear {N}'.format(N=len(r))]

        for i,j in sim.ensemble.pairs:
            #tabulate, drop entries where r = 0
            u = np.delete(sim.potentials.pair.energy((i,j)),k)
            f = np.delete(sim.potentials.pair.force((i,j)),k)

            # write and read potential data
            _file = sim.directory.file('pair.{i}.{j}.dat'.format(i=i,j=j))
            cmds += ['pair_write {i} {j} {N} r {inner} {outer} pair.{i}.{j}.dat TABLE_{i}_{j}'.format(i=i,
                                                                                                      j=i,
                                                                                                      N=len(r),
                                                                                                      inner=r[0],
                                                                                                      outer=r[-1]),
                    'pair_coeff {i} {j} pair.{i}.{j}.dat TABLE_{i}_{j}'.format(i=i,j=j)]

        return cmds

class InitializeFromFile(Initialize):
    pass

class InitializeRandomly(Initialize):
    def __init__(self, neighbor_buffer, seed, units='lj', atom_style='atomic'):
        super().__init__(neighbor_buffer)
        self.seed = seed
        self.units = units
        self.atom_style = atom_style

    def to_commands(self, sim):
        cmds = ['units {style}'.format(style=self.units),
                'boundary p p p',
                'atom_style {style}'.format(style=self.atom_style)]

        # make box from ensemble
        box = self.extract_box_params(sim)
        if not np.all(np.isclose(box[-3:],0)):
            cmds += ['region box prism {} {} {} {} {} {} {} {} {}'.format(*box)]
        else:
            cmds += ['region box block {} {} {} {} {} {}'.format(*box[:-3])]
        cmds += ['create_box {N} box'.format(N=len(sim.ensemble.types))]

        # use lammps random initialization routines
        for i in sim.ensemble.types:
            cmds += ['create_atoms {typeid} random {N} {seed} box'.format(typeid=sim.type_map[i],
                                                                          N=sim.ensemble.N[i],
                                                                          seed=self.seed+sim.type_map[i]-1)]
        cmds += ['mass * 1.0',
                 'velocity all create {temp} {seed}'.format(temp=sim.ensemble.T,seed=self.seed)]

        cmds += self.attach_potentials(sim)

        return cmds

class MinimizeEnergy(simulate.SimulationOperation):
    pass
class AddMDIntegrator(simulate.SimulationOperation):
    pass
class RemoveMDIntegrator(simulate.SimulationOperation):
    pass
class AddBrownianIntegrator(AddMDIntegrator):
    pass
class RemoveBrownianIntegrator(RemoveMDIntegrator):
    pass
class AddLangevinIntegrator(AddMDIntegrator):
    pass
class RemoveLangevinIntegrator(RemoveMDIntegrator):
    pass
class AddNPTIntegrator(AddMDIntegrator):
    pass
class RemoveNPTIntegrator(RemoveMDIntegrator):
    pass
class AddNVTIntegrator(AddMDIntegrator):
    pass
class RemoveNVTIntegrator(RemoveMDIntegrator):
    pass
class Run(simulate.SimulationOperation):
    pass
class RunUpTo(simulate.SimulationOperation):
    pass
class AddEnsembleAnalyzer(simulate.SimulationOperation):
    pass
