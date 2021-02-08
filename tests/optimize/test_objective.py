"""Unit tests for objective module."""
import unittest

import relentless

class QuadraticObjective(relentless.optimize.ObjectiveFunction):
    """Mock objective function used to test relentless.optimize.ObjectiveFunction"""

    def __init__(self, x):
        self.x = x

    def run(self):
        val = self.x.value**2
        grad = {self.x:2*self.x.value}
        return relentless.optimize.Result(val, grad)

    def design_variables(self):
        return (self.x,)

class test_ObjectiveFunction(unittest.TestCase):
    """Unit tests for relentless.optimize.ObjectiveFunction"""

    def test_run(self):
        """Test run method"""
        x = relentless.variable.DesignVariable(value=3.0)
        q = QuadraticObjective(x=x)

        res = q.run()
        self.assertAlmostEqual(res.value, 9.0)
        self.assertAlmostEqual(res.gradient[x], 6.0)
