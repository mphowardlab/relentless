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

        #test using line search for step size
        l = relentless.optimize.LineSearch(abs_tol=1e-8, max_step_size=0.25, max_iter=100)
        o.step_size = l
        self.assertDictEqual(o.abs_tol, {x:1e-9})
        self.assertEqual(o.step_size, l)
        self.assertEqual(o.max_iter, 1000)

        #test invalid parameters
        with self.assertRaises(ValueError):
            o.step_size = -0.25
        with self.assertRaises(ValueError):
            o.max_iter = 0
        with self.assertRaises(TypeError):
            o.max_iter = 100.0
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
        o.max_iter = 1
        self.assertFalse(o.optimize(objective=q))

        #test using line search for step size
        x.value = 0.0
        o.max_iter = 1000
        l = relentless.optimize.LineSearch(abs_tol=1e-8, max_step_size=25, max_iter=99)
        o.step_size = l
        #self.assertTrue(o.optimize(objective=q))
        self.assertAlmostEqual(x.value, 1.0)

class test_LineSearch(unittest.TestCase):
    """Unit tests for relentless.optimize.LineSearch"""

    def test_init(self):
        """Test creation with data."""
        x = relentless.variable.DesignVariable(value=3.0)
        q = QuadraticObjective(x=x)

        l = relentless.optimize.LineSearch(abs_tol=1e-8, max_step_size=1, max_iter=1000)
        self.assertAlmostEqual(l.abs_tol, 1e-8)
        self.assertAlmostEqual(l.max_step_size, 1)
        self.assertEqual(l.max_iter, 1000)

        #test invalid parameters
        with self.assertRaises(ValueError):
            l.max_step_size = -0.25
        with self.assertRaises(ValueError):
            l.max_iter = 0
        with self.assertRaises(TypeError):
            l.max_iter = 100.0
        with self.assertRaises(ValueError):
            l.abs_tol = -1e-9
        with self.assertRaises(TypeError):
            l.abs_tol = {x:1e-9}

    def test_find(self):
        """Test find method."""
        x = relentless.variable.DesignVariable(value=3.0)
        q = QuadraticObjective(x=x)

        #without passing in initial result
        l = relentless.optimize.LineSearch(abs_tol=1e-8, max_step_size=25, max_iter=1000)
        step = l.find(objective=q)
        self.assertGreater(step, 0)
        self.assertLess(step, 25)

        #with passing in initial result
        x.value = 99.9
        res = q.compute()
        l = relentless.optimize.LineSearch(abs_tol=1e-8, max_step_size=100, max_iter=1000)
        step = l.find(objective=q, result_0=res)
        self.assertGreater(step, 0)
        self.assertLess(step, 100)

        #max step size can be acceptable
        x.value = -1.0
        l = relentless.optimize.LineSearch(abs_tol=1e-8, max_step_size=0.01, max_iter=1000)
        step = l.find(objective=q)
        self.assertEqual(step, 0.01)
