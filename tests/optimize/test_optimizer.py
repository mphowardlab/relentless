"""Unit tests for optimizer module."""
import unittest

import relentless

from .test_objective import QuadraticObjective

class test_Optimizer(unittest.TestCase):
    """Unit tests for relentless.optimize.Optimizer"""

    def test_run(self):
        """Test run method."""
        x = relentless.variable.DesignVariable(value=3.0)
        q = QuadraticObjective(x=x)
        o = relentless.optimize.SteepestDescent(obj=q, alpha=0.25)

        o.run()
        self.assertAlmostEqual(x.value, 4.5)

        o.run()
        self.assertAlmostEqual(x.value, 6.75) 
