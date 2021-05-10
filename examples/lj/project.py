"""Optimize the energy parameter of the Lennard-Jones potential.

This script demonstrates optimization of the :math:`\varepsilon` parameter
of the Lennard-Jones potential using a dilute simulation where the pair
distribution function is known exactly.

For demonstration purposes, the target ensemble is generated from the standard
Lennard-Jones potential with :math:`\varepsilon = 1.0` and :math:`T = 1.5`. The
value of :math:`\varepsilon` is then changed to 1.5, and the optimization
"rediscovers" its correct value.

The optimization method is a fixed-step descent with a line search, which should
converges with a small number of iterations.

"""
import numpy as np
import relentless

# lj potential with epsilon as a design variable
lj = relentless.potential.LennardJones(types=('1',))
epsilon = relentless.variable.DesignVariable(1.0)
lj.coeff['1','1'].update({'epsilon': epsilon, 'sigma': 1.0, 'rmax': 3.0, 'shift': True})

# target ensemble
target = relentless.ensemble.Ensemble(T=1.5, V=relentless.volume.Cube(L=10.), N={'1':50})
dr = 0.05
rs = np.arange(0.5*dr,5.0,dr)
for pair in target.pairs:
    gs = np.exp(-target.beta*lj.energy(pair,rs))
    target.rdf[pair] = relentless.ensemble.RDF(rs,gs)

# relative entropy optimization in dilute molecular simulation
potentials = relentless.simulate.Potentials(lj)
potentials.pair.rmax = 3.0
potentials.pair.num = 1000
thermo = relentless.simulate.AddEnsembleAnalyzer(check_thermo_every=1, check_rdf_every=1, rdf_dr=dr)
simulation = relentless.simulate.Dilute([thermo])
relent = relentless.optimize.RelativeEntropy(target, simulation, potentials, thermo)

# steepest descent optimization (fixed step with line search)
optimizer = relentless.optimize.FixedStepDescent(stop=relentless.optimize.GradientTest(1e-4),
                                                 max_iter=10,
                                                 step_size=0.1,
                                                 line_search=relentless.optimize.LineSearch(1e-6,2))

# change parameter and optimize
epsilon.value = 1.5
optimizer.optimize(relent)
print('Optimized epsilon = {:.3f}'.format(epsilon.value))
