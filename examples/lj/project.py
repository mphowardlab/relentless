import numpy as np

import relentless

class Desktop(relentless.environment.Environment):
    mpiexec = 'mpirun -np {np}'
    always_wrap = False

# lj potential
lj = relentless.potential.LJPotential(types=('1',), shift=True)
lj.coeff['1','1'] = {'epsilon': 1.0, 'sigma': 0.9, 'rmax': lambda c : 3*c['1','1']['sigma']}

# reference ensemble
tgt = relentless.ensemble.NVT(N={'1': 50}, V=1000., T=1.5)
dr = 0.1
rs = np.arange(0.5*dr,5.0,dr)
tgt.rdf['1','1'] = np.column_stack((rs,np.exp(-tgt.beta*lj(rs,('1','1')))))

# change parameters and setup optimization
lj.coeff['1','1'] = {'epsilon': 1.0, 'sigma': 1.0}
lj.variables['1','1'] = (relentless.Variable('sigma', low=0.8, high=1.2),)

# mock simulation engine & rdf (dilute limit)
sim = relentless.engine.LAMMPS(lammps='lmp_mpi', template='nvt.in', policy=relentless.environment.Policy(procs=2))
rdf = relentless.rdf.LAMMPS(ensemble=tgt, order={'T': 1, 'P': 2, 'V': 3, 'N_1': 4})
tab = relentless.potential.Tabulator(nbins=1000, rmin=0.0, rmax=3.6, fmax=100., fcut=1.e-6)

# relative entropy
re = relentless.optimize.RelativeEntropy(tgt, sim, rdf, tab)
re.add_potential(lj)

# steepest descent
opt = relentless.optimize.GradientDescent(re)

with Desktop(path='./workspace') as env:
    opt.run(env=env, maxiter=10, rates={('1','1'): 1e-3})
