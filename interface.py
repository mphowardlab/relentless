import numpy as np

import relentless

# reference ensemble
tgt = relentless.ensemble.NVT(N={'1': 10, '2': 10}, V=1000., T=1.0)
# lj potential
lj = relentless.potential.LJPotential(types=('1','2'), shift=True)
lj.coeff['1','1'] = {'epsilon': 1.0, 'sigma': 0.9, 'rmax': lambda c : 3*c['1','1']['sigma']}
lj.coeff['2','2'] = {'epsilon': 1.0, 'sigma': 1.1, 'rmax': lambda c : 3*c['2','2']['sigma']}
lj.coeff['1','2'] = {'epsilon': lambda c : 0.5*(c['1','1']['epsilon']+c['2','2']['epsilon']),
                     'sigma': lambda c : 0.5*(c['1','1']['sigma']+c['2','2']['sigma']),
                     'rmax': lambda c : 3*c['1','2']['sigma'](c)}
# rdf with current values
dr = 0.1
rs = np.arange(0.5*dr,5.0,dr)
tgt.rdf['1','1'] = np.column_stack((rs,np.exp(-tgt.beta*lj(rs,('1','1')))))
tgt.rdf['2','2'] = np.column_stack((rs,np.exp(-tgt.beta*lj(rs,('2','2')))))
tgt.rdf['1','2'] = np.column_stack((rs,np.exp(-tgt.beta*lj(rs,('1','2')))))

# change parameters and setup optimization
lj.coeff['1','1'] = {'epsilon': 1.0, 'sigma': 1.0}
lj.coeff['2','2'] = {'epsilon': 1.0, 'sigma': 1.0}
lj.variables['1','1'] = (relentless.Variable('sigma', low=0.1),)
lj.variables['2','2'] = (relentless.Variable('sigma', low=0.1),)

# mock simulation engine & rdf (dilute limit)
sim = relentless.engine.LAMMPS(ensemble=tgt, lammps='lmp_mpi', template='nvt.in')
traj = relentless.trajectory.Mock(ensemble=tgt)
rdf = relentless.rdf.Mock(dr=0.1, potential=lj)

tab = relentless.potential.Tabulator(nbins=1000, rmin=0.0, rmax=4.0, fmax=100., fcut=1.e-6)
opt = relentless.optimize.SteepestDescent(sim, traj, rdf, tab)
opt.add_potential(lj)

with relentless.environment.Lonestar(path='./build', mock=True) as env:
    opt.run(env=env, target=tgt, maxiter=50, rates={('1','1'): 1e-2,
                                                    ('2','2'): 1e-2})
