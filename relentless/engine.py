import numpy as np

from .environment import Policy

class Engine(object):
    trim = False

    def __init__(self, policy):
        self.policy = policy

    def run(self, env, step, potential):
        """ Run the simulation with the current potentials and return the trajectory.
        """
        raise NotImplementedError()

class Mock(Engine):
    def __init__(self, policy=Policy()):
        super().__init__(policy)

    def run(self, env, step, potentials):
        pass

class LAMMPS(Engine):
    trim = True

    def __init__(self, lammps, template, args=None, policy=Policy()):
        super().__init__(policy)

        self.lammps = lammps
        self.template = template
        self.args = args if args is not None else ""

    def run(self, env, step, potentials):
        with env.data(step):
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
