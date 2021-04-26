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
