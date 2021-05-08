import numpy as np

import relentless

# lj potential appended to tabulated potentials
lj = relentless.potential.LennardJones(types=('1',))
epsilon = relentless.variable.DesignVariable(1.0)
sigma = relentless.variable.DesignVariable(value=0.9, low=0.8, high=1.2)
lj.coeff['1','1'].update({'epsilon': epsilon, 'sigma': sigma, 'rmax': 2.7})
potentials = relentless.simulate.Potentials(lj)
potentials.pair.rmax = 3.6
potentials.pair.num = 1000

# target ensemble
target = relentless.ensemble.Ensemble(T=1.5, V=relentless.volume.Cube(L=10.), N={'1':50})
dr = 0.1
rs = np.arange(0.5*dr,5.0,dr)
gs = np.exp(-target.beta*lj.energy(('1','1'),rs))
target.rdf['1','1'] = relentless.ensemble.RDF(rs,gs)

# dilute molecular simulation
thermo = relentless.simulate.dilute.AddEnsembleAnalyzer()
simulation = relentless.simulate.dilute.Dilute([thermo])

# relative entropy + steepest descent
relent = relentless.optimize.RelativeEntropy(target, simulation, potentials, thermo)
tol = relentless.optimize.GradientTest(1e-4)
optimizer = relentless.optimize.SteepestDescent(tol, max_iter=1000, step_size=0.005)

# change parameters and optimize
epsilon.value = 0.5
sigma.value = 1.1
optimizer.optimize(relent)
