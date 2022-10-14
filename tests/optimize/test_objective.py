"""Unit tests for objective module."""
import json
import tempfile
import unittest

import numpy
import scipy.integrate

import relentless

class QuadraticObjective(relentless.optimize.ObjectiveFunction):
    """Mock objective function used to test relentless.optimize.ObjectiveFunction"""

    def __init__(self, x):
        self.x = x

    def compute(self, variables, directory=None):
        val = (self.x.value-1)**2
        variables = relentless.variable.graph.check_variables_and_types(variables, relentless.variable.Variable)
        grad = {}
        for x in variables:
            if x is self.x:
                grad[x] = 2*(self.x.value-1)
            else:
                grad[x] = 0.

        # optionally write output
        if directory is not None:
            with open(directory.file('x.log'),'w') as f:
                f.write(str(self.x.value) + '\n')

        return relentless.optimize.ObjectiveFunctionResult(variables, val, grad, directory)

class test_ObjectiveFunction(unittest.TestCase):
    """Unit tests for relentless.optimize.ObjectiveFunction"""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()

    def test_compute(self):
        """Test compute method"""
        x = relentless.variable.DesignVariable(value=4.0)
        q = QuadraticObjective(x=x)

        res = q.compute(x)
        self.assertAlmostEqual(res.value, 9.0)
        self.assertAlmostEqual(res.gradient[x], 6.0)
        self.assertAlmostEqual(res.variables[x], 4.0)

        x.value = 3.0
        self.assertAlmostEqual(res.variables[x], 4.0) # maintains the value at time of construction

        # test "invalid" variable
        with self.assertRaises(KeyError):
            m = res.gradient[relentless.variable.SameAs(x)]

    def test_directory(self):
        x = relentless.variable.DesignVariable(value=1.0)
        q = QuadraticObjective(x=x)
        d = relentless.data.Directory(self.directory.name)
        res = q.compute(x, d)

        with open(d.file('x.log')) as f:
            x = float(f.readline())
        self.assertAlmostEqual(x,1.0)

    def tearDown(self):
        self.directory.cleanup()
        del self.directory

class test_RelativeEntropy(unittest.TestCase):
    """Unit tests for relentless.optimize.RelativeEntropy"""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()

        lj = relentless.potential.LennardJones(types=('1',))
        self.epsilon = relentless.variable.DesignVariable(value=1.0)
        self.sigma = relentless.variable.DesignVariable(value=0.9)
        lj.coeff['1','1'].update({'epsilon':self.epsilon, 'sigma':self.sigma, 'rmax':2.7})
        self.potentials = relentless.simulate.Potentials(pair_potentials=lj)
        self.potentials.pair.rmax = 3.6
        self.potentials.pair.num = 1000
        self.potentials.pair.fmax = 100.
        self.potentials.pair.neighbor_buffer = 0.4

        v_obj = relentless.extent.Cube(L=10.)
        self.target = relentless.ensemble.Ensemble(T=1.5, V=v_obj, N={'1':50})
        rs = numpy.arange(0.05,5.0,0.1)
        gs = numpy.exp(-lj.energy(('1','1'),rs))
        self.target.rdf['1','1'] = relentless.ensemble.RDF(r=rs, g=gs)

        init = relentless.simulate.InitializeRandomly(seed=42, N=self.target.N, V=self.target.V, T=self.target.T)
        self.thermo = relentless.simulate.AddEnsembleAnalyzer(
            check_thermo_every=1, check_rdf_every=1, rdf_dr=0.1)
        self.simulation = relentless.simulate.dilute.Dilute(init, operations=[self.thermo])

    def relent_grad(self, var, ext=False):
        rs = numpy.linspace(0,3.6,1001)[1:]
        r6_inv = numpy.power(0.9/rs, 6)
        gs = numpy.exp(-(1/1.5)*4.*1.0*(r6_inv**2 - r6_inv))
        sim_rdf = relentless.math.Interpolator(rs,gs)

        rs = numpy.arange(0.05,5.0,0.1)
        r6_inv = numpy.power(0.9/rs, 6)
        gs = numpy.exp(-4.*1.0*(r6_inv**2 - r6_inv))
        tgt_rdf = relentless.math.Interpolator(rs,gs)

        rs = numpy.linspace(0,3.6,1001)[1:]
        r6_inv = numpy.power(0.9/rs, 6)
        if var is self.epsilon:
            dus = 4*(r6_inv**2 - r6_inv)
        elif var is self.sigma:
            dus = (48.*1.0/0.9)*(r6_inv**2 - 0.5*r6_inv)
        dudvar = relentless.math.Interpolator(rs,dus)

        if ext:
            norm_factor = 1.
        else:
            norm_factor = 1000.
        sim_factor = 50**2*(1/1.5)/(1000*norm_factor)
        tgt_factor = 50**2*(1/1.5)/(1000*norm_factor)

        r = numpy.linspace(0.05, 3.55, 1001)
        y = -2*numpy.pi*r**2*(sim_factor*sim_rdf(r)-tgt_factor*tgt_rdf(r))*dudvar(r)
        return scipy.integrate.trapz(x=r, y=y)

    def test_init(self):
        relent = relentless.optimize.RelativeEntropy(self.target,
                                                     self.simulation,
                                                     self.potentials,
                                                     self.thermo)
        self.assertEqual(relent.target, self.target)
        self.assertEqual(relent.simulation, self.simulation)
        self.assertEqual(relent.potentials, self.potentials)
        self.assertEqual(relent.thermo, self.thermo)

        # test invalid target ensemble
        with self.assertRaises(ValueError):
            relent.target = relentless.ensemble.Ensemble(T=1.5, P=1, N={'1':50})

    def test_compute(self):
        """Test compute and compute_gradient methods"""
        relent = relentless.optimize.RelativeEntropy(self.target,
                                                     self.simulation,
                                                     self.potentials,
                                                     self.thermo)

        res = relent.compute((self.epsilon, self.sigma))

        sim = self.simulation.run(self.potentials, self.directory.name)
        ensemble = self.thermo.extract_ensemble(sim)
        res_grad = relent.compute_gradient(ensemble, (self.epsilon, self.sigma))

        grad_eps = self.relent_grad(self.epsilon)
        grad_sig = self.relent_grad(self.sigma)

        self.assertIsNone(res.value)
        numpy.testing.assert_allclose(res.gradient[self.epsilon], grad_eps, atol=1e-4)
        numpy.testing.assert_allclose(res.gradient[self.sigma], grad_sig, atol=1e-4)
        numpy.testing.assert_allclose(res_grad[self.epsilon], grad_eps, atol=1e-4)
        numpy.testing.assert_allclose(res_grad[self.sigma], grad_sig, atol=1e-4)

        # test extensive option
        relent = relentless.optimize.RelativeEntropy(self.target,
                                                     self.simulation,
                                                     self.potentials,
                                                     self.thermo,
                                                     extensive=True)

        res = relent.compute((self.epsilon,self.sigma))

        sim = self.simulation.run(self.potentials, self.directory.name)
        ensemble = self.thermo.extract_ensemble(sim)
        res_grad = relent.compute_gradient(ensemble, (self.epsilon, self.sigma))

        grad_eps = self.relent_grad(self.epsilon, ext=True)
        grad_sig = self.relent_grad(self.sigma, ext=True)

        self.assertIsNone(res.value)
        numpy.testing.assert_allclose(res.gradient[self.epsilon], grad_eps, atol=1e-1)
        numpy.testing.assert_allclose(res.gradient[self.sigma], grad_sig, atol=1e-1)
        numpy.testing.assert_allclose(res_grad[self.epsilon], grad_eps, atol=1e-1)
        numpy.testing.assert_allclose(res_grad[self.sigma], grad_sig, atol=1e-1)

    def test_directory(self):
        relent = relentless.optimize.RelativeEntropy(self.target,
                                                     self.simulation,
                                                     self.potentials,
                                                     self.thermo)

        d = relentless.data.Directory(self.directory.name)
        res = relent.compute((self.epsilon,self.sigma),d)

        with open(d.file('pair_potential.0.json')) as f:
            x = json.load(f)
        self.assertAlmostEqual(x["('1', '1')"]['epsilon'], self.epsilon.value)
        self.assertAlmostEqual(x["('1', '1')"]['sigma'], self.sigma.value)
        self.assertAlmostEqual(x["('1', '1')"]['rmax'], 2.7)

        sim = self.simulation.run(self.potentials, self.directory.name)
        sim_ens = self.thermo.extract_ensemble(sim)
        with open(d.file('ensemble.json')) as g:
            y = json.load(g)
        self.assertAlmostEqual(y['T'], 1.5)
        self.assertAlmostEqual(y['N'], {'1':50})
        self.assertAlmostEqual(y['V']['data'], {'L':10.})
        self.assertAlmostEqual(y['P'], sim_ens.P)
        numpy.testing.assert_allclose(y['rdf']["('1', '1')"], sim_ens.rdf['1','1'].table.tolist())

    def tearDown(self):
        self.directory.cleanup()
        del self.directory

if __name__ == '__main__':
    unittest.main()
