import numpy as np

import relentless

class Desktop(relentless.environment.Environment):
    mpiexec = 'mpirun -np {np}'
    always_wrap = False

# lj potential
lj = relentless.potential.LJPotential(types=('1','2'), shift=True)
lj.coeff['1','1'] = {'epsilon': 1.0, 'sigma': 0.9, 'rmax': lambda c : 3*c['1','1']['sigma']}
lj.coeff['2','2'] = {'epsilon': 1.0, 'sigma': 1.1, 'rmax': lambda c : 3*c['2','2']['sigma']}
lj.coeff['1','2'] = {'epsilon': 1.0,
                     'sigma': lambda c : 0.5*(c['1','1']['sigma']+c['2','2']['sigma']),
                     'rmax': lambda c : 3*c['1','2']['sigma'](c)}

# reference ensemble
tgt = relentless.ensemble.NVT(N={'1': 50, '2': 50}, V=1000., T=1.5)
dr = 0.1
rs = np.arange(0.5*dr,5.0,dr)
for pair in tgt.rdf:
    tgt.rdf[pair] = np.column_stack((rs,np.exp(-tgt.beta*lj(rs,pair))))

# change parameters and setup optimization
lj.coeff['1','1'] = {'epsilon': 1.0, 'sigma': 1.0}
lj.coeff['2','2'] = {'epsilon': 1.0, 'sigma': 1.0}
lj.variables['1','1'] = (relentless.Variable('sigma', low=0.8, high=1.2),)
lj.variables['2','2'] = (relentless.Variable('sigma', low=0.8, high=1.2),)

# mock simulation engine & rdf (dilute limit)
sim = relentless.engine.LAMMPS(lammps='lmp_mpi', template='nvt.in', policy=relentless.environment.Policy(procs=2))
rdf = relentless.rdf.LAMMPS(ensemble=tgt, order={'T': 1, 'P': 2, 'V': 3, 'N_1': 4, 'N_2': 5})
tab = relentless.potential.Tabulator(nbins=1000, rmin=0.0, rmax=3.6, fmax=100., fcut=1.e-6)

# relative entropy
re = relentless.optimize.RelativeEntropy(tgt, sim, rdf, tab)
re.add_potential(lj)

# steepest descent
opt = relentless.optimize.GradientDescent(re)

with Desktop(path='./workspace') as env:
    opt.run(env=env, maxiter=10, rates={('1','1'): 1e-3, ('2','2'): 1e-3})
