"""Unit tests for objective module."""
import json
import tempfile
import unittest

import numpy as np
import scipy.integrate

import relentless

class QuadraticObjective(relentless.optimize.ObjectiveFunction):
    """Mock objective function used to test relentless.optimize.ObjectiveFunction"""

    def __init__(self, x):
        self.x = x

    def compute(self, directory=None):
        val = (self.x.value-1)**2
        grad = {self.x:2*(self.x.value-1)}

        # optionally write output
        if directory is not None:
            with open(directory.file('x.log'),'w') as f:
                f.write(str(self.x.value) + '\n')

        res = self.make_result(val, grad, directory)
        return res

    def design_variables(self):
        return (self.x,)

class test_ObjectiveFunction(unittest.TestCase):
    """Unit tests for relentless.optimize.ObjectiveFunction"""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()

    def test_compute(self):
        """Test compute method"""
        x = relentless.variable.DesignVariable(value=4.0)
        q = QuadraticObjective(x=x)

        res = q.compute()
        self.assertAlmostEqual(res.value, 9.0)
        self.assertAlmostEqual(res.gradient[x], 6.0)
        self.assertCountEqual(res.design_variables.todict().keys(), q.design_variables())

        x.value = 3.0
        self.assertDictEqual(res.design_variables.todict(), {x: 4.0}) #maintains the value at time of construction

        #test "invalid" variable
        with self.assertRaises(KeyError):
            m = res.gradient[relentless.variable.SameAs(x)]

    def test_design_variables(self):
        """Test design_variables method"""
        x = relentless.variable.DesignVariable(value=1.0)
        q = QuadraticObjective(x=x)

        self.assertEqual(q.x.value, 1.0)
        self.assertCountEqual((x,), q.design_variables())

        x.value = 3.0
        self.assertEqual(q.x.value, 3.0)
        self.assertCountEqual((x,), q.design_variables())

    def test_directory(self):
        x = relentless.variable.DesignVariable(value=1.0)
        q = QuadraticObjective(x=x)
        d = relentless.data.Directory(self.directory.name)
        res = q.compute(d)

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

        self.dr = 0.1

        lj = relentless.potential.LennardJones(types=('1',))
        self.epsilon = relentless.variable.DesignVariable(value=1.0)
        self.sigma = relentless.variable.DesignVariable(value=0.9)
        lj.coeff['1','1'].update({'epsilon':self.epsilon, 'sigma':self.sigma, 'rmax':2.7})
        self.potentials = relentless.simulate.Potentials(pair_potentials=lj)
        self.potentials.pair.rmax = 3.6
        self.potentials.pair.num = 1000
        self.potentials.pair.fmax = 100.

        v_obj = relentless.volume.Cube(L=10.)
        self.target = relentless.ensemble.Ensemble(T=1.5, V=v_obj, N={'1':50})
        rs = np.arange(0.5*self.dr, 5.0, self.dr)
        gs = np.exp(-self.target.beta*lj.energy(('1','1'),rs))
        self.target.rdf['1','1'] = relentless.ensemble.RDF(r=rs, g=gs)

        self.thermo = relentless.simulate.dilute.AddEnsembleAnalyzer()
        self.simulation = relentless.simulate.dilute.Dilute(operations=[self.thermo])

    def test_init(self):
        relent = relentless.optimize.RelativeEntropy(self.target,
                                                     self.simulation,
                                                     self.potentials,
                                                     self.thermo,
                                                     self.dr)
        self.assertEqual(relent.target, self.target)
        self.assertEqual(relent.simulation, self.simulation)
        self.assertEqual(relent.potentials, self.potentials)
        self.assertEqual(relent.thermo, self.thermo)
        self.assertAlmostEqual(relent.dr, self.dr)
        self.assertIsNone(relent.communicator)

        #test communicator argument
        comm = relentless.mpi.world
        relent.communicator = comm
        self.assertEqual(relent.target, self.target)
        self.assertEqual(relent.simulation, self.simulation)
        self.assertEqual(relent.potentials, self.potentials)
        self.assertEqual(relent.thermo, self.thermo)
        self.assertAlmostEqual(relent.dr, self.dr)
        self.assertEqual(relent.communicator, comm)

        #test invalid target ensemble
        with self.assertRaises(ValueError):
            relent.target = relentless.ensemble.Ensemble(T=1.5, P=1, N={'1':50})

    def relent_grad(self, var):
        sim = self.simulation.run(self.target, self.potentials, directory=None)
        sim_ens = self.thermo.extract_ensemble(sim)
        rs = self.potentials.pair.r[1:]
        dus = self.potentials.pair.derivative(('1','1'),var)[1:]
        dudvar = relentless._math.Interpolator(rs,dus)
        sim_factor = sim_ens.N['1']**2*sim_ens.beta/(sim_ens.V.volume*self.target.V.volume)
        tgt_factor = self.target.N['1']**2*self.target.beta/(self.target.V.volume**2)

        r = np.arange(0.05, 3.55, self.dr)
        y = 2*np.pi*r**2*(sim_factor*sim_ens.rdf['1','1'](r)-tgt_factor*self.target.rdf['1','1'](r))*dudvar(r)
        return scipy.integrate.trapz(x=r, y=y)

    def test_compute(self):
        """Test compute method"""
        relent = relentless.optimize.RelativeEntropy(self.target,
                                                     self.simulation,
                                                     self.potentials,
                                                     self.thermo,
                                                     self.dr)

        grad_eps = self.relent_grad(self.epsilon)
        grad_sig = self.relent_grad(self.sigma)

        res = relent.compute()
        self.assertIsNone(res.value)
        self.assertAlmostEqual(res.gradient[self.epsilon], grad_eps)
        self.assertAlmostEqual(res.gradient[self.sigma], grad_sig)
        self.assertCountEqual(res.design_variables, (self.epsilon,self.sigma))

    def test_design_variables(self):
        """Test design_variables method"""
        relent = relentless.optimize.RelativeEntropy(self.target,
                                                     self.simulation,
                                                     self.potentials,
                                                     self.thermo,
                                                     self.dr)

        self.assertCountEqual((self.epsilon,self.sigma), relent.design_variables())

        #test constant variable
        self.epsilon.const = True
        self.assertCountEqual((self.sigma,), relent.design_variables())

        self.sigma.const = True
        self.assertCountEqual((), relent.design_variables())

    def test_directory(self):
        relent = relentless.optimize.RelativeEntropy(self.target,
                                                     self.simulation,
                                                     self.potentials,
                                                     self.thermo,
                                                     self.dr)

        d = relentless.data.Directory(self.directory.name)
        res = relent.compute(d)

        with open(d.file('potential_0.log')) as f:
            x = json.load(f)
        self.assertAlmostEqual(x["('1', '1')"]['epsilon'], self.epsilon.value)
        self.assertAlmostEqual(x["('1', '1')"]['sigma'], self.sigma.value)
        self.assertAlmostEqual(x["('1', '1')"]['rmax'], 2.7)

    def tearDown(self):
        self.directory.cleanup()
        del self.directory

if __name__ == '__main__':
    unittest.main()
