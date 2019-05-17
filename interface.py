import numpy as np

import relentless

# reference ensemble
tgt = relentless.ensemble.NVT(N={'A': 10, 'B': 10}, V=1000., T=1.0)
# lj potential
lj = relentless.potential.LJPotential(types=('A','B'), shift=True)
lj.coeff['A','A'] = {'epsilon': 0.9, 'sigma': 0.9, 'rmax': lambda c : 3*c['A','A']['sigma']}
lj.coeff['B','B'] = {'epsilon': 1.1, 'sigma': 1.5, 'rmax': lambda c : 3*c['B','B']['sigma']}
lj.coeff['A','B'] = {'epsilon': lambda c : 0.5*(c['A','A']['epsilon']+c['B','B']['epsilon']),
                     'sigma': lambda c : 0.5*(c['A','A']['sigma']+c['B','B']['sigma']),
                     'rmax': lambda c : 3*c['A','B']['sigma'](c)}
# rdf with current values
dr = 0.1
rs = np.arange(0.5*dr,5.0,dr)
tgt.rdf['A','A'] = np.column_stack((rs,np.exp(-tgt.beta*lj(rs,('A','A')))))
tgt.rdf['B','B'] = np.column_stack((rs,np.exp(-tgt.beta*lj(rs,('B','B')))))
tgt.rdf['A','B'] = np.column_stack((rs,np.exp(-tgt.beta*lj(rs,('A','B')))))

# change parameters and setup optimization
lj.coeff['A','A'] = {'epsilon': 1.0, 'sigma': 1.0}
lj.coeff['B','B'] = {'epsilon': 1.0, 'sigma': 1.0}
lj.variables['A','A'] = (relentless.Variable('epsilon', low=0.0),
                         relentless.Variable('sigma', low=0.1))
lj.variables['B','B'] = (relentless.Variable('epsilon', low=0.0),
                         relentless.Variable('sigma', low=0.1))

# mock simulation engine & rdf (dilute limit)
sim = relentless.engine.Mock()
traj = relentless.trajectory.Mock(ensemble=tgt)
rdf = relentless.rdf.Mock(dr=0.1, potential=lj)

tab = relentless.potential.Tabulator(nbins=1000, rmin=0.0, rmax=10.0, fmax=100., fcut=1.e-6)
opt = relentless.optimize.SteepestDescent(sim, traj, rdf, tab)
opt.add_potential(lj)

with relentless.environment.Lonestar(path='./build', mock=True) as env:
    opt.run(env=env, target=tgt, maxiter=50, rates={('A','A'): 1.5e-2,
                                                    ('B','B'): 1.5e-2})
