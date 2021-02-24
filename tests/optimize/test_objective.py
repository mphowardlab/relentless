"""Unit tests for objective module."""
import unittest

import relentless

class QuadraticObjective(relentless.optimize.ObjectiveFunction):
    """Mock objective function used to test relentless.optimize.ObjectiveFunction"""

    def __init__(self, x):
        self.x = x

    def compute(self):
        val = (self.x.value-1)**2
        grad = {self.x:2*(self.x.value-1)}

        res = self.make_result(val, grad)
        return res

    def design_variables(self):
        return (self.x,)

class test_ObjectiveFunction(unittest.TestCase):
    """Unit tests for relentless.optimize.ObjectiveFunction"""

    def test_compute(self):
        """Test compute method"""
        x = relentless.variable.DesignVariable(value=4.0)
        q = QuadraticObjective(x=x)

        res = q.compute()
        self.assertAlmostEqual(res.value, 9.0)
        self.assertAlmostEqual(res.gradient(x), 6.0)
        self.assertCountEqual(res.design_variables(), q.design_variables())

        #test "invalid" variable
        p = relentless.variable.SameAs(x)
        self.assertEqual(res.gradient(p), 0.0)

    def test_design_variables(self):
        """Test design_variables method"""
        x = relentless.variable.DesignVariable(value=1.0)
        q = QuadraticObjective(x=x)

        self.assertEqual(q.x.value, 1.0)
        self.assertCountEqual((x,), q.design_variables())

        x.value = 3.0
        self.assertEqual(q.x.value, 3.0)
        self.assertCountEqual((x,), q.design_variables())
