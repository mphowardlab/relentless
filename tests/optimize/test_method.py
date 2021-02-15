"""Unit tests for method module."""
import unittest

import relentless

from .test_objective import QuadraticObjective

class test_SteepestDescent(unittest.TestCase):
    """Unit tests for relentless.optimize.SteepestDescent"""

    def test_run(self):
        """Test run method."""
        x = relentless.variable.DesignVariable(value=3.0)
        q = QuadraticObjective(x=x)
        o = relentless.optimize.SteepestDescent(alpha=0.25)

        o.optimize(objective=q)
        self.assertAlmostEqual(x.value, 1.0)
