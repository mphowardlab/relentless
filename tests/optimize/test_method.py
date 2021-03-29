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
        o = relentless.optimize.SteepestDescent(abs_tol=1e-8, max_iter=1000, step_size=0.25)
        self.assertAlmostEqual(o.abs_tol, 1e-8)
        self.assertEqual(o.max_iter, 1000)
        self.assertAlmostEqual(o.step_size, 0.25)
        self.assertIsNone(o.line_search)

        #test dictionary of tolerances
        o.abs_tol = {x:1e-9}
        self.assertDictEqual(o.abs_tol, {x:1e-9})
        self.assertEqual(o.max_iter, 1000)
        self.assertAlmostEqual(o.step_size, 0.25)
        self.assertIsNone(o.line_search)

        #test using line search for step size
        l = relentless.optimize.LineSearch(abs_tol=1e-8, max_iter=100)
        o.line_search = l
        self.assertDictEqual(o.abs_tol, {x:1e-9})
        self.assertEqual(o.max_iter, 1000)
        self.assertEqual(o.step_size, 0.25)
        self.assertEqual(o.line_search, l)

        #test invalid parameters
        with self.assertRaises(ValueError):
            o.abs_tol = -1e-9
        with self.assertRaises(ValueError):
            o.abs_tol = {x:-1e-10}
        with self.assertRaises(ValueError):
            o.max_iter = 0
        with self.assertRaises(TypeError):
            o.max_iter = 100.0
        with self.assertRaises(ValueError):
            o.step_size = -0.25
        with self.assertRaises(TypeError):
            o.line_search = q

    def test_run(self):
        """Test run method."""
        x = relentless.variable.DesignVariable(value=3.0)
        q = QuadraticObjective(x=x)
        o = relentless.optimize.SteepestDescent(abs_tol=1e-8, max_iter=1000, step_size=0.25)

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

        #test using line search option
        x.value = 3
        o.max_iter = 1000
        l = relentless.optimize.LineSearch(abs_tol=1e-8, max_iter=100)
        o.line_search = l
        self.assertTrue(o.optimize(objective=q))
        self.assertAlmostEqual(x.value, 1.0)

class test_LineSearch(unittest.TestCase):
    """Unit tests for relentless.optimize.LineSearch"""

    def test_init(self):
        """Test creation with data."""
        x = relentless.variable.DesignVariable(value=3.0)
        q = QuadraticObjective(x=x)

        l = relentless.optimize.LineSearch(abs_tol=1e-8, max_iter=1000)
        self.assertAlmostEqual(l.abs_tol, 1e-8)
        self.assertEqual(l.max_iter, 1000)

        #test invalid parameters
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
        l = relentless.optimize.LineSearch(abs_tol=1e-8, max_iter=1000)
        x = relentless.variable.DesignVariable(value=-3.0)
        q = QuadraticObjective(x=x)
        res_1 = q.compute()

        #bracketing the minimum (find step size that takes function to minimum)
        x.value = 3.0
        res_2 = q.compute()
        x.value = -3.0
        res_new = l.find(objective=q, start=res_1, end=res_2)
        self.assertAlmostEqual(res_new.design_variables[x], 1.0)
        self.assertAlmostEqual(res_new.gradient[x], 0.0)
        self.assertEqual(q.x.value, -3.0)

        #not bracketing the minimum (accept "maximum" step size)
        x.value = -1.0
        res_3 = q.compute()
        x.value = -3.0
        res_new = l.find(objective=q, start=res_1, end=res_3)
        self.assertAlmostEqual(res_new.design_variables[x], -1.0)
        self.assertAlmostEqual(res_new.gradient[x], -4.0)
        self.assertEqual(q.x.value, -3.0)

        #bound does not include current objective value
        res_new = l.find(objective=q, start=res_3, end=res_2)
        self.assertAlmostEqual(res_new.design_variables[x], 1.0)
        self.assertAlmostEqual(res_new.gradient[x], 0.0)
        self.assertEqual(q.x.value, -3.0)

        #invalid search interval (not descent direction)
        with self.assertRaises(ValueError):
            res_new = l.find(objective=q, start=res_3, end=res_1)
