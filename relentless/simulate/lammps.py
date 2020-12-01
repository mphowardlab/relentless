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
    def __init__(self, operations, quiet=True, **options):
        if not _lammps_found:
            raise ImportError('LAMMPS not found.')

        self.quiet = quiet
        super().__init__(operations,**options)

    def _new_instance(self, ensemble, potentials, directory):
        sim = super()._new_instance(ensemble,potentials,directory)

        if self.quiet:
            # create lammps instance with all output disabled
            launch_args = ['-echo','none',
                           '-log','none',
                           '-screen','none',
                           '-nocite']
        else:
            launch_args = ['-echo','screen',
                           '-log', sim.directory.file('log.lammps'),
                           '-nocite']
        sim.lammps = lammps.lammps(cmdargs=launch_args)

        # lammps uses 1-indexed ints for types, so build mapping in both direction
        sim.type_map = {}
        sim.typeid_map = {}
        for i,t in enumerate(sim.ensemble.types):
            sim.type_map[t] = i+1
            sim.typeid_map[sim.type_map[t]] = t

        return sim

class LAMMPSOperation(simulate.SimulationOperation):
    fix_id = 1

    def __call__(self, sim):
        cmds = self.to_commands(sim)
        sim.lammps.commands_list(cmds)

    @classmethod
    def add_fix(cls):
        LAMMPSOperation.fix_id += 1
        return LAMMPSOperation.fix_id

    @abc.abstractmethod
    def to_commands(self, sim):
        pass

class Initialize(LAMMPSOperation):
    def __init__(self, neighbor_buffer, units='lj', atom_style='atomic'):
        self.neighbor_buffer = neighbor_buffer
        self.units = units
        self.atom_style = atom_style

    def to_commands(self, sim):
        cmds = ['units {style}'.format(style=self.units),
                'boundary p p p',
                'atom_style {style}'.format(style=self.atom_style)]

        return cmds

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
        # lammps requires r > 0
        flags = sim.potentials.pair.r > 0
        r = sim.potentials.pair.r[flags]
        Nr = len(r)
        if Nr == 1:
            raise ValueError('LAMMPS requires at least two points in the tabulated potential.')

        # check that all r are equally spaced
        dr = r[1:]-r[:-1]
        if not np.all(np.isclose(dr,dr[0])):
            raise ValueError('LAMMPS requires equally spaced r in pair potentials.')

        def pair_map(sim,pair):
            # Map lammps type indexes as a pair, lowest type first
            i,j = pair
            id_i = sim.type_map[i]
            id_j = sim.type_map[j]
            if id_i > id_j:
                id_i,id_j = id_j,id_i

            return id_i,id_j

        # write all potentials into a file
        file_ = sim.directory.file('lammps_pair_table.dat')
        with open(file_,'w') as fw:
            fw.write('# LAMMPS tabulated pair potentials\n')
            for i,j in sim.ensemble.pairs:
                id_i,id_j = pair_map(sim,(i,j))
                fw.write(('# pair ({i},{j})\n'
                          '\n'
                          'TABLE_{id_i}_{id_j}\n').format(i=i,
                                                          j=j,
                                                          id_i=id_i,
                                                          id_j=id_j)
                        )
                fw.write('N {N} R {rmin} {rmax}\n\n'.format(N=Nr,
                                                            rmin=r[0],
                                                            rmax=r[-1]))

                u = sim.potentials.pair.energy((i,j))[flags]
                f = sim.potentials.pair.force((i,j))[flags]
                for idx in range(Nr):
                    fw.write('{idx} {r} {u} {f}\n'.format(idx=idx+1,r=r[idx],u=u[idx],f=f[idx]))

        # process all lammps commands
        cmds = ['neighbor {skin} multi'.format(skin=self.neighbor_buffer)]
        cmds += ['pair_style table linear {N}'.format(N=Nr)]
        for i,j in sim.ensemble.pairs:
            # get lammps type indexes, lowest type first
            id_i,id_j = pair_map(sim,(i,j))
            cmds += ['pair_coeff {id_i} {id_j} {filename} TABLE_{id_i}_{id_j}'.format(id_i=id_i,id_j=id_j,filename=file_)]

        return cmds

class InitializeFromFile(Initialize):
    def __init__(self, filename, neighbor_buffer, units='lj', atom_style='atomic'):
        super().__init__(neighbor_buffer, units, atom_style)
        self.filename = filename

    def to_commands(self, sim):
        cmds = super().to_commands(sim)
        cmds += ['read_data {filename}'.format(filename=self.filename)]
        cmds += self.attach_potentials(sim)

        return cmds

class InitializeRandomly(Initialize):
    def __init__(self, neighbor_buffer, seed, units='lj', atom_style='atomic'):
        super().__init__(neighbor_buffer, units, atom_style)
        self.seed = seed

    def to_commands(self, sim):
        cmds = super().to_commands(sim)

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
                 'velocity all create {temp} {seed}'.format(temp=sim.ensemble.T,
                                                            seed=self.seed)]

        cmds += self.attach_potentials(sim)

        return cmds

class MinimizeEnergy(LAMMPSOperation):
    def __init__(self, energy_tolerance, force_tolerance, max_iterations, dt):
        self.energy_tolerance = energy_tolerance
        self.force_tolerance = force_tolerance
        self.max_iterations = max_iterations

    def to_commands(self, sim):
        cmds = ['minimize {etol} {ftol} {maxiter} {maxeval}'.format(etol=self.energy_tolerance,
                                                                    ftol=self.force_tolerance,
                                                                    maxiter=self.max_iterations,
                                                                    maxeval=self.max_iterations)]

        return cmds

# Brownian dynamics not supported by LAMMPS

class AddLangevinIntegrator(LAMMPSOperation):
    def __init__(self, dt, friction, seed):
        self.friction = friction
        self.seed = seed

        self.lgv_idx = super().add_fix()
        self.nve_idx = super().add_fix()

    def to_commands(self, sim):
        cmds = ['fix {idx} {group_idx} langevin {t_start} {t_stop} {damp} {seed}'.format(idx=self.lgv_idx,
                                                                                         group_idx='all',
                                                                                         t_start=sim.ensemble.T,
                                                                                         t_stop=sim.ensemble.T,
                                                                                         damp=self.friction,
                                                                                         seed=self.seed),
                'fix {idx} {group_idx} nve'.format(idx=self.nve_idx,
                                                   group_idx='all')]

        return cmds

class RemoveLangevinIntegrator(LAMMPSOperation):
    def __init__(self, add_op):
        if not isinstance(add_op, AddLangevinIntegrator):
            raise TypeError('Addition operation is not AddLangevinIntegrator.')
        self.add_op = add_op

    def to_commands(self, sim):
        cmds = ['unfix {idx}'.format(idx=self.add_op.lgv_idx),
                'unfix {idx}'.format(idx=self.add_op.nve_idx)]

        return cmds

class AddNPTIntegrator(LAMMPSOperation):
    def __init__(self, dt, tau_T, tau_P):
        self.tau_T = tau_T
        self.tau_P = tau_P

        self.idx = super().add_fix()

    def to_commands(self, sim):
        cmds = ['fix {idx} {group_idx} npt temp {Tstart} {Tstop} {Tdamp} iso {Pstart} {Pstop} {Pdamp}'.format(idx=self.idx,
                                                                                                              group_idx='all',
                                                                                                              Tstart = sim.ensemble.T,
                                                                                                              Tstop = sim.ensemble.T,
                                                                                                              Tdamp = self.tau_T,
                                                                                                              Pstart = sim.ensemble.P,
                                                                                                              Pstop = sim.ensemble.P,
                                                                                                              Pdamp = self.tau_P)]

        return cmds

class RemoveNPTIntegrator(LAMMPSOperation):
    def __init__(self, add_op):
        if not isinstance(add_op, AddNPTIntegrator):
            raise TypeError('Addition operation is not AddNPTIntegrator.')
        self.add_op = add_op

    def to_commands(self, sim):
        cmds = ['unfix {idx}'.format(idx=self.add_op.idx)]

        return cmds

class AddNVTIntegrator(LAMMPSOperation):
    def __init__(self, dt, tau_T):
        self.tau_T = tau_T

        self.idx = super().add_fix()

    def to_commands(self, sim):
        cmds = ['fix {idx} {group_idx} nvt temp {Tstart} {Tstop} {Tdamp}'.format(idx=self.idx,
                                                                                 group_idx='all',
                                                                                 Tstart = sim.ensemble.T,
                                                                                 Tstop = sim.ensemble.T,
                                                                                 Tdamp = self.tau_T)]

        return cmds

class RemoveNVTIntegrator(LAMMPSOperation):
    def __init__(self, add_op):
        if not isinstance(add_op, AddNVTIntegrator):
            raise TypeError('Addition operation is not AddNVTIntegrator.')
        self.add_op = add_op

    def to_commands(self, sim):
        cmds = ['unfix {idx}'.format(idx=self.add_op.idx)]

        return cmds

class Run(LAMMPSOperation):
    def __init__(self, steps):
        self.steps = steps

    def to_commands(self, sim):
        cmds = ['run {N}'.format(N=self.steps)]

        return cmds

class RunUpTo(LAMMPSOperation):
    def __init__(self, step):
        self.step = step

    def to_commands(self, sim):
        cmds = ['run {N} upto'.format(N=self.step)]

        return cmds

class AddEnsembleAnalyzer(LAMMPSOperation):
    pass
