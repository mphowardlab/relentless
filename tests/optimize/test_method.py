"""Unit tests for method module."""
import unittest

import relentless

from .test_objective import QuadraticObjective

class test_SteepestDescent(unittest.TestCase):
    """Unit tests for relentless.optimize.SteepestDescent"""

    def test_init(self):
        """Test creation with data."""
        x = relentless.variable.DesignVariable(value=3.0)
        q = QuadraticObjective(x=x)

        #test scalar tolerance
        o = relentless.optimize.SteepestDescent(abs_tol=1e-8, step_size=0.25, max_iter=1000)
        self.assertAlmostEqual(o.abs_tol, 1e-8)
        self.assertAlmostEqual(o.step_size, 0.25)
        self.assertEqual(o.max_iter, 1000)

        #test dictionary of tolerances
        o.abs_tol = {x:1e-9}
        self.assertDictEqual(o.abs_tol, {x:1e-9})
        self.assertAlmostEqual(o.step_size, 0.25)
        self.assertEqual(o.max_iter, 1000)

        #test invalid parameters
        with self.assertRaises(ValueError):
            o.step_size = -0.25
        with self.assertRaises(ValueError):
            o.max_iter = 0
        with self.assertRaises(ValueError):
            o.abs_tol = -1e-9
        with self.assertRaises(ValueError):
            o.abs_tol = {x:-1e-10}

    def test_run(self):
        """Test run method."""
        x = relentless.variable.DesignVariable(value=3.0)
        q = QuadraticObjective(x=x)
        o = relentless.optimize.SteepestDescent(abs_tol=1e-8, step_size=0.25, max_iter=1000)

        self.assertTrue(o.optimize(objective=q))
        self.assertAlmostEqual(x.value, 1.0)

        #test dictionary of tolerances
        x.value = -9.81
        o.abs_tol = {x:1e-9}
        self.assertTrue(o.optimize(objective=q))
        self.assertAlmostEqual(x.value, 1.0)

        #test insufficient maximum iterations
        x.value = 1.5
        o = relentless.optimize.SteepestDescent(abs_tol=1e-8, step_size=0.25, max_iter=1)
        self.assertFalse(o.optimize(objective=q))
