import gzip
import os

import numpy as np

from . import core
from . import environment
from . import utils

class Engine(object):
    trim = False

    def __init__(self, policy):
        self.policy = policy

    def setup(self, step):
        pass

    def run(self, env, step, potential):
        """ Run the simulation with the current potentials and return the trajectory.
        """
        raise NotImplementedError()

    def load_trajectory(self, env):
        raise NotImplementedError()

class MockEngine(Engine):
    def __init__(self, ensemble, policy=environment.Policy()):
        super(MockEngine, self).__init__(policy)
        self.ensemble = ensemble

    def run(self, env, step, potentials):
        pass

    def load_trajectory(self, env):
        L = self.ensemble.V**(1./3.)
        box = core.Box(L)

        Ntot = np.sum([self.ensemble.N[i] for i in self.ensemble.types])
        snap = core.Snapshot(Ntot, box)
        first = 0
        for i in self.ensemble.types:
            last = first + self.ensemble.N[i]
            snap.types[first:last] = i
            first += self.ensemble.N[i]

        return core.Trajectory(snapshots=(snap,))

class LAMMPS(Engine):
    trim = True

    def __init__(self, lammps, template, args=None, policy=environment.Policy()):
        super(LAMMPS, self).__init__(policy)

        self.lammps = lammps
        self.template = template
        self.args = args if args is not None else ""

    def run(self, env, step, potentials):
        assert env.cwd is not None, 'Environment current working directory must be set.'
        with env.cwd:
            for i,j in potentials:
                if j >= i:
                    pot = np.asarray(potentials[(i,j)])
                    assert pot.shape[1] == 3, 'Potential must be given as (r,u,f) pairs'

                    file_ = 'pair.{i}.{j}.dat'.format(s=step,i=i,j=j)
                    with open(file_,'w') as fw:
                        fw.write(('# Tabulated pair for ({i},{j}) at step = {s}\n'
                                  '\n'
                                  'TABLE_{i}_{j}\n').format(s=step,i=i,j=j))
                        fw.write('N {N} R {rmin} {rmax}\n\n'.format(N=pot.shape[0],rmin=pot[0,0],rmax=pot[-1,0]))
                        for idx,(r,u,f) in enumerate(pot):
                            fw.write('{idx} {r} {u} {f}\n'.format(idx=idx,r=r,u=u,f=f))

            # inject the tabulated pair potentials into the template
            file_ = self.template

            # run the simulation out of the scratch directory
            cmd = '{lmp} -in {fn} -nocite {args}'.format(lmp=self.lammps, fn=file_, args=self.args)
            env.call(cmd, self.policy)

    def load_trajectory(self, env):
        assert env.cwd is not None, 'Environment current working directory must be set.'

        traj = core.Trajectory()
        with env.cwd:
            file_ = 'trajectory.gz'
            if os.path.exists(file_):
                gz = True
            else:
                file_ = 'trajectory'
                gz = False

        return traj
