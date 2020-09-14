import jinja2
import numpy as np

from relentless.core import RDF
from relentless.engine import Engine
from relentless.environment import Policy

class LAMMPS(Engine):
    trim = False
    default_options = {'dt': 0.005, 'eq': 1000000, 'sample': 1000000, 'dump': 50, 'seed': 42}

    def __init__(self, ensemble, table, template, lammps='lmp_mpi', args=None, options=None, policy=Policy(), potentials=None):
        super().__init__(ensemble, table, potentials)

        self.template = template
        self.lammps = lammps
        self.args = args if args is not None else ""
        self.policy = policy

        # load up options, including user overrides
        self.options = dict(self.default_options)
        if options is not None:
            for key,value in options.items():
                self.options[key] = value

    def run(self, env, step):
        if len(self.potentials) == 0:
            raise RuntimeError('Cannot run simulation without potentials.')

        with env.data(step):
            table_size = 0
            file_pairs = {}

            potentials = self._tabulate_potentials()
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
                f.write(template.render(size=table_size, files=file_pairs, ensemble=self.ensemble, options=self.options))

            # run the simulation
            cmd = '{lmp} -in {fn} -screen none -nocite {args}'.format(lmp=self.lammps, fn=self.template, args=self.args)
            result = env.call(cmd, self.policy)

    def process(self, env, step):
        ens = self.ensemble.copy()

        with env.data(step):
            # thermodynamic property accumulators
            T = 0.
            P = 0.
            V = 0.
            with open('log.lammps') as f:
                # advance to the table section
                line = f.readline()
                while line and 'RELENTLESS PRODUCTION' not in line:
                    line = f.readline()
                while line and 'Step' not in line:
                    line = f.readline()

                header = line.strip().split()
                n_entry = len(header)
                num_samples = 0
                line = f.readline()
                while line and 'Loop time' not in line:
                    row = line.strip().split()
                    if len(row) != n_entry:
                        raise IOError('Read bad row of LAMMPS thermo data')

                    T += float(row[1])
                    P += float(row[2])
                    V += float(row[3])
                    num_samples += 1

                    line = f.readline()

                # normalize mean
                if num_samples > 0:
                    T /= num_samples
                    P /= num_samples
                    V /= num_samples
                else:
                    raise IOError('LAMMPS thermo table empty!')

            # set conjugate variables as needed for ensemble
            ens.T = T
            if 'P' in ens.conjugates:
                ens.P = P
            if 'V' in ens.conjugates:
                ens.V = V

            # read radial distribution functions from file
            rdf = np.loadtxt('rdf.lammps', skiprows=4)
            rs = rdf[:,1]
            for i,pair in enumerate(ens.rdf):
                ens.rdf[pair] = RDF(rs,rdf[:,2+2*i])

        return ens
