"""Unit tests for objective module."""
import tempfile
import unittest

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

    def test_compute(self):
        """Test compute method"""
        v_obj = relentless.volume.Cube(L=3.0)
        target = relentless.ensemble.Ensemble(T=10, V=v_obj, N={'A':1, 'B':2})

        thermo = relentless.simulate.dilute.AddEnsembleAnalyzer()
        operations = [relentless.simulate.InitializeRandomly(seed=2,neighbor_buffer=0.4),
                      relentless.simulate.AddNVTIntegrator(dt=0.1,tau_T=1.0),
                      thermo]
        simulation = relentless.simulate.dilute.Dilute(operations)

        lj = relentless.potential.LennardJones(types=('1','2'))
        potentials = relentless.simulate.Potentials(pair_potentials=lj)
        potentials.pair.rmax = 5.0
        potentials.pair.num = 6

        relent = relentless.optimize.RelativeEntropy(target,simulation,potentials,thermo,dr=0.2)
        res = relent.compute()

if __name__ == '__main__':
    unittest.main()
