"""Unit tests for method module."""
import os
import tempfile
import unittest

import relentless

from .test_objective import QuadraticObjective

class test_LineSearch(unittest.TestCase):
    """Unit tests for relentless.optimize.LineSearch"""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()

    def test_init(self):
        """Test creation with data."""
        x = relentless.variable.DesignVariable(value=3.0)
        q = QuadraticObjective(x=x)

        l = relentless.optimize.LineSearch(tolerance=1e-8, max_iter=1000)
        self.assertAlmostEqual(l.tolerance, 1e-8)
        self.assertEqual(l.max_iter, 1000)

        #test invalid parameters
        with self.assertRaises(ValueError):
            l.max_iter = 0
        with self.assertRaises(TypeError):
            l.max_iter = 100.0

    def test_find(self):
        """Test find method."""
        l = relentless.optimize.LineSearch(tolerance=1e-8, max_iter=1000)
        x = relentless.variable.DesignVariable(value=-3.0)
        q = QuadraticObjective(x=x)
        res_1 = q.compute(x)

        #bracketing the minimum (find step size that takes function to minimum)
        x.value = 3.0
        res_2 = q.compute(x)
        x.value = -3.0
        res_new = l.find(objective=q, start=res_1, end=res_2)
        self.assertAlmostEqual(res_new.design_variables[x], 1.0)
        self.assertAlmostEqual(res_new.gradient[x], 0.0)
        self.assertEqual(q.x.value, -3.0)

        # use directory for output, check result of first iteration
        d = relentless.data.Directory(self.directory.name)
        res_new = l.find(q, res_1, res_2, d)
        self.assertAlmostEqual(res_new.design_variables[x], 1.0)
        self.assertAlmostEqual(res_new.gradient[x], 0.0)
        self.assertEqual(q.x.value, -3.0)
        self.assertTrue(os.path.isdir(os.path.join(d.path,'0')))
        self.assertTrue(os.path.isfile(os.path.join(d.path,'0','x.log')))

        #not bracketing the minimum (accept "maximum" step size)
        x.value = -1.0
        res_3 = q.compute(x)
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

        #invalid search interval (0 distance from start to end)
        with self.assertRaises(ValueError):
            res_new = l.find(objective=q, start=res_3, end=res_3)

        #invalid tolerance
        with self.assertRaises(ValueError):
            l.tolerance = -1e-9
            l.find(objective=q, start=res_1, end=res_3)
        with self.assertRaises(ValueError):
            l.tolerance = 1.1
            l.find(objective=q, start=res_1, end=res_3)

    def tearDown(self):
        self.directory.cleanup()
        del self.directory

class test_SteepestDescent(unittest.TestCase):
    """Unit tests for relentless.optimize.SteepestDescent"""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()

    def test_init(self):
        """Test creation with data."""
        x = relentless.variable.DesignVariable(value=3.0)
        q = QuadraticObjective(x=x)
        t = relentless.optimize.GradientTest(tolerance=1e-8)

        o = relentless.optimize.SteepestDescent(stop=t, max_iter=1000, step_size=0.25)
        self.assertEqual(o.stop, t)
        self.assertEqual(o.max_iter, 1000)
        self.assertAlmostEqual(o.step_size, 0.25)
        self.assertAlmostEqual(o.scale, 1.0)
        self.assertIsNone(o.line_search)

        #test scalar scaling parameter
        o.scale = 0.5
        self.assertEqual(o.stop, t)
        self.assertEqual(o.max_iter, 1000)
        self.assertAlmostEqual(o.step_size, 0.25)
        self.assertAlmostEqual(o.scale, 0.5)
        self.assertIsNone(o.line_search)

        #test dictionary of scaling parameters
        o.scale = {x:0.3}
        self.assertEqual(o.stop, t)
        self.assertEqual(o.max_iter, 1000)
        self.assertAlmostEqual(o.step_size, 0.25)
        self.assertDictEqual(o.scale, {x:0.3})
        self.assertIsNone(o.line_search)

        #test using line search
        l = relentless.optimize.LineSearch(tolerance=1e-9, max_iter=100)
        o.line_search = l
        self.assertEqual(o.stop, t)
        self.assertEqual(o.max_iter, 1000)
        self.assertAlmostEqual(o.step_size, 0.25)
        self.assertDictEqual(o.scale, {x:0.3})
        self.assertEqual(o.line_search, l)

        #test invalid parameters
        with self.assertRaises(TypeError):
            o.stop = 1e-8
        with self.assertRaises(ValueError):
            o.max_iter = 0
        with self.assertRaises(TypeError):
            o.max_iter = 100.0
        with self.assertRaises(ValueError):
            o.step_size = -0.25
        with self.assertRaises(ValueError):
            o.scale = -0.5
        with self.assertRaises(ValueError):
            o.scale = {x:-0.5}
        with self.assertRaises(TypeError):
            o.line_search = q

    def test_run(self):
        """Test run method."""
        x = relentless.variable.DesignVariable(value=3.0)
        q = QuadraticObjective(x=x)
        t = relentless.optimize.GradientTest(tolerance=1e-8)
        o = relentless.optimize.SteepestDescent(stop=t, max_iter=1000, step_size=0.25)

        self.assertTrue(o.optimize(objective=q, design_variables=x))
        self.assertAlmostEqual(x.value, 1.0)

        #test insufficient maximum iterations
        x.value = 1.5
        o.max_iter = 1
        self.assertFalse(o.optimize(objective=q, design_variables=x))

        #test with nontrivial scalar scaling parameter
        x.value = 50
        o.scale = 0.85
        o.max_iter = 1000
        self.assertTrue(o.optimize(objective=q, design_variables=x))
        self.assertAlmostEqual(x.value, 1.0)

        #test with nontrivial dictionary of scaling parameters
        x.value = -35
        o.scale = {x:1.5}
        self.assertTrue(o.optimize(objective=q, design_variables=x))
        self.assertAlmostEqual(x.value, 1.0)

        #test using line search option
        x.value = 3
        o.line_search = relentless.optimize.LineSearch(tolerance=1e-5, max_iter=100)
        self.assertTrue(o.optimize(objective=q, design_variables=x))
        self.assertAlmostEqual(x.value, 1.0)

    def test_directory(self):
        x = relentless.variable.DesignVariable(value=1.5)
        q = QuadraticObjective(x=x)
        t = relentless.optimize.GradientTest(tolerance=1e-8)
        o = relentless.optimize.SteepestDescent(stop=t, max_iter=1, step_size=0.25)

        # optimize with output
        d = relentless.data.Directory(self.directory.name)
        o.optimize(q, x, d)

        # 0/ holds the initial value
        self.assertTrue(os.path.isdir(os.path.join(d.path,'0')))
        self.assertTrue(os.path.isfile(os.path.join(d.path,'0','x.log')))
        with open(d.directory('0').file('x.log')) as f:
            self.assertAlmostEqual(float(f.readline()), 1.5)

        # 0/.next should be empty because it has been accepted to 1/
        self.assertTrue(os.path.isdir(os.path.join(d.path,'0','.next')))
        self.assertEqual(len(os.listdir(d.directory('0/.next').path)), 0)

        # 1/ holds the next output
        self.assertTrue(os.path.isdir(os.path.join(d.path,'1')))
        self.assertTrue(os.path.isfile(os.path.join(d.path,'1','x.log')))
        with open(d.directory('1').file('x.log')) as f:
            self.assertAlmostEqual(float(f.readline()), 1.25)

    def test_directory_line_search(self):
        x = relentless.variable.DesignVariable(value=0.5)
        q = QuadraticObjective(x=x)
        t = relentless.optimize.GradientTest(tolerance=1e-8)
        o = relentless.optimize.SteepestDescent(stop=t, max_iter=1, step_size=2.)
        o.line_search = relentless.optimize.LineSearch(tolerance=1e-5, max_iter=1)

        # optimize with output
        d = relentless.data.Directory(self.directory.name)
        o.optimize(q, x, d)

        # 0/ holds the initial value
        self.assertTrue(os.path.isdir(os.path.join(d.path,'0')))
        with open(d.directory('0').file('x.log')) as f:
            self.assertAlmostEqual(float(f.readline()), 0.5)

        # 0/.next/ holds the overshoot
        self.assertTrue(os.path.isdir(os.path.join(d.path,'0','.next')))
        with open(d.directory('0/.next').file('x.log')) as f:
            self.assertAlmostEqual(float(f.readline()), 2.5)

        # 0/.line/ should exist, but it should have only one entry (0/) because
        # line search is exact for this function
        self.assertTrue(os.path.isdir(os.path.join(d.path,'0','.line')))
        self.assertEqual(len(os.listdir(d.directory('0/.line').path)), 1)

        # .line/0 should be empty because it has been accepted to 1/
        self.assertTrue(os.path.isdir(os.path.join(d.path,'0','.line','0')))
        self.assertEqual(len(os.listdir(d.directory('0/.line/0').path)), 0)

        # 1/ holds the solved value
        self.assertTrue(os.path.isdir(os.path.join(d.path,'1')))
        with open(d.directory('1').file('x.log')) as f:
            self.assertAlmostEqual(float(f.readline()), 1.0)

    def tearDown(self):
        self.directory.cleanup()
        del self.directory

class test_FixedStepDescent(unittest.TestCase):
    """Unit tests for relentless.optimize.FixedStepDescent"""

    def test_run(self):
        """Test run method."""
        x = relentless.variable.DesignVariable(value=3.0)
        q = QuadraticObjective(x=x)
        t = relentless.optimize.GradientTest(tolerance=1e-8)
        o = relentless.optimize.FixedStepDescent(stop=t, max_iter=1000, step_size=0.25)

        self.assertTrue(o.optimize(objective=q, design_variables=x))
        self.assertAlmostEqual(x.value, 1.0)

        #test insufficient maximum iterations
        x.value = 1.5
        o.max_iter = 1
        self.assertFalse(o.optimize(objective=q, design_variables=x))

        #test step size that does not converge
        x.value = 1.5
        o.step_size = 0.42
        o.max_iter = 10000
        self.assertFalse(o.optimize(objective=q, design_variables=x))

        #test with nontrivial scalar scaling parameter
        x.value = 50
        o.step_size = 0.25
        o.scale = 4.0
        self.assertTrue(o.optimize(objective=q, design_variables=x))
        self.assertAlmostEqual(x.value, 1.0)

        #test with nontrivial dictionary of scaling parameters
        x.value = -35
        o.scale = {x:1.5}
        self.assertTrue(o.optimize(objective=q, design_variables=x))
        self.assertAlmostEqual(x.value, 1.0)

        #test using line search option
        x.value = 3
        o.line_search = relentless.optimize.LineSearch(tolerance=1e-5, max_iter=100)
        self.assertTrue(o.optimize(objective=q, design_variables=x))
        self.assertAlmostEqual(x.value, 1.0)

if __name__ == '__main__':
    unittest.main()
