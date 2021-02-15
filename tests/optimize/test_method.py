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
        o = relentless.optimize.SteepestDescent(alpha=0.25, max_iter=1000, abs_tol=1e-8, mode='grad_diff')

        o.optimize(objective=q)
        self.assertAlmostEqual(x.value, 1.0)

        x.value = -9.81
        o.optimize(objective=q)
        self.assertAlmostEqual(x.value, 1.0)

        #test insufficient maximum iterations
        x.value = 1.5
        o = relentless.optimize.SteepestDescent(alpha=0.25, max_iter=1, abs_tol=1e-8, mode='grad_diff')
        with self.assertRaises(RuntimeError):
            o.optimize(objective=q)

        #test invalid convergence mode
        o = relentless.optimize.SteepestDescent(alpha=0.25, max_iter=1000, abs_tol=1e-8, mode='percent_error')
        with self.assertRaises(ValueError):
            o.optimize(objective=q)
