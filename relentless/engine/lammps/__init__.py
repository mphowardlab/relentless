import numpy as np
import jinja2

from relentless.engine import Engine
from relentless.environment import Policy

class LAMMPS(Engine):
    trim = False

    def __init__(self, lammps, template, args=None, policy=Policy()):
        super().__init__(policy)

        self.lammps = lammps
        self.template = template
        self.args = args if args is not None else ""

    def run(self, env, step, ensemble, potentials):
        with env.data(step):
            table_size = 0
            file_pairs = {}

            for i,j in potentials:
                if j >= i:
                    pot = np.asarray(potentials[(i,j)])
                    assert pot.shape[1] == 3, 'Potential must be given as (r,u,f) pairs'
                    # drop zero from first entry
                    if pot[0,0] <= 0.0:
                        pot = pot[1:]
                    # check table size
                    if table_size == 0:
                        table_size = pot.shape[0]
                    elif pot.shape[0] != table_size:
                        raise IndexError('LAMMPS requires equal sized tables.')

                    file_ = 'pair.{i}.{j}.dat'.format(s=step,i=i,j=j)
                    file_pairs[(i,j)] = file_

                    with open(file_,'w') as fw:
                        fw.write(('# Tabulated pair for ({i},{j}) at step = {s}\n'
                                  '\n'
                                  'TABLE_{i}_{j}\n').format(s=step,i=i,j=j))
                        fw.write('N {N} R {rmin} {rmax}\n\n'.format(N=pot.shape[0],rmin=pot[0,0],rmax=pot[-1,0]))
                        for idx,(r,u,f) in enumerate(pot):
                            fw.write('{idx} {r} {u} {f}\n'.format(idx=idx+1,r=r,u=u,f=f))

            # inject the tabulated pair potentials into the template
            loader = jinja2.Environment(
                loader=jinja2.ChoiceLoader([
                    jinja2.FileSystemLoader(env.project.path),
                    jinja2.PackageLoader('relentless.engine.lammps')]),
                undefined=jinja2.StrictUndefined,
                trim_blocks=True)

            template = loader.get_template(self.template)
            with open(self.template, 'w') as f:
                f.write(template.render(size=table_size, files=file_pairs, ensemble=ensemble))

            # run the simulation out of the scratch directory
            cmd = '{lmp} -in {fn} -screen none -nocite {args}'.format(lmp=self.lammps, fn=self.template, args=self.args)
            result = env.call(cmd, self.policy)
