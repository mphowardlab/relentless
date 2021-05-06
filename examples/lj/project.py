import numpy as np

import relentless

# lj potential appended to tabulated potentials
lj = relentless.potential.LennardJones(types=('1',))
epsilon = relentless.variable.DesignVariable(value=1.0)
sigma = relentless.variable.DesignVariable(value=0.9)
lj.coeff['1','1'].update({'epsilon': epsilon, 'sigma': sigma, 'rmax': 2.7})
potentials = relentless.simulate.Potentials(pair_potentials=lj)
potentials.pair.rmax = 3.6
potentials.pair.num = 1000
potentials.pair.fmax = 100.

# target ensemble
target = relentless.ensemble.Ensemble(T=1.5, V=relentless.volume.Cube(L=10.), N={'1':50})
dr = 0.1
rs = np.arange(0.5*dr,5.0,dr)
target.rdf['1','1'] = relentless.ensemble.RDF(rs,np.exp(-target.beta*lj.energy(('1','1'),rs)))

# change parameters for optimization
epsilon.value = 1.0
sigma.value = 1.0
sigma.low = 0.8
sigma.high = 1.2

# dilute molecular simulation
thermo = relentless.simulate.dilute.AddEnsembleAnalayzer()
simulation = relentless.simulate.dilute.Dilute(operations=[thermo])

# relative entropy + steepest descent
relent = relentless.optimize.RelativeEntropy(target, simulation, potentials, thermo)
tol = relentless.optimize.GradientTest(tolerance=1e-8)
optimizer = relentless.optimize.SteepestDescent(stop=tol, max_iter=1000, step_size=0.25)

optimizer.optimize(objective=relent)
