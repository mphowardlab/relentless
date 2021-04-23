"""Unit tests for criteria module."""
import unittest

import relentless

from .test_objective import QuadraticObjective

class test_Tolerance(unittest.TestCase):
    """Unit tests for relentless.optimize.Tolerance"""

    def test_init(self):
        """Test creation with data."""
        x = relentless.variable.DesignVariable(value=3.0)

        #test scalar tolerances
        t = relentless.optimize.Tolerance(absolute=1.0, relative=0.5)
        self.assertAlmostEqual(t.absolute, 1.0)
        self.assertAlmostEqual(t.relative, 0.5)
        self.assertAlmostEqual(t._atol(x), 1.0)
        self.assertAlmostEqual(t._rtol(x), 0.5)

        #test dictionaries of tolerances
        t.absolute = {x:1.1}
        t.relative = {x:0.6}
        self.assertDictEqual(t.absolute, {x:1.1})
        self.assertDictEqual(t.relative, {x:0.6})
        self.assertAlmostEqual(t._atol(x), 1.1)
        self.assertAlmostEqual(t._rtol(x), 0.6)

        #test invalid tolerances
        with self.assertRaises(ValueError):
            t.absolute = -1e-8
        with self.assertRaises(ValueError):
            t.absolute = {x:-1e-8}
        with self.assertRaises(ValueError):
            t.relative = 1.2
        with self.assertRaises(ValueError):
            t.relative = {x:-0.1}

    def test_isclose(self):
        """Test isclose method."""
        x = relentless.variable.DesignVariable(value=3.0)

        #test scalar tolerances
        t = relentless.optimize.Tolerance(absolute=0.1, relative=0.1)
        self.assertFalse(t.isclose(x, x.value, 2.5))
        self.assertTrue(t.isclose(x, x.value, 2.7))

        #test dictionary of tolerances
        t.absolute = {x:0.2}
        t.relative = {x:0.2}
        self.assertFalse(t.isclose(x, x.value, 2.2))
        self.assertTrue(t.isclose(x, x.value, 2.4))

class test_GradientTest(unittest.TestCase):
    """Unit tests for relentless.optimize.GradientTest"""

    def test_init(self):
        """Test creation with data."""
        x = relentless.variable.DesignVariable(value=3.0)

        #test scalar tolerance
        t = relentless.optimize.GradientTest(tolerance=1e-8)
        self.assertAlmostEqual(t.tolerance, 1e-8)

        #test dictionary of tolerances
        t.tolerance = {x:1e-8}
        self.assertDictEqual(t.tolerance, {x:1e-8})

    def test_converged(self):
        """Test converged method."""
        x = relentless.variable.DesignVariable(value=3.0)
        q = QuadraticObjective(x=x)

        #test scalar tolerance
        t = relentless.optimize.GradientTest(tolerance=1e-8)
        self.assertFalse(t.converged(result=q.compute()))
        x.value = 0.999999999
        self.assertTrue(t.converged(result=q.compute()))

        #test dictionary of tolerances
        x.value = 3.0
        t.tolerance = {x:1e-9}
        self.assertFalse(t.converged(result=q.compute()))
        x.value = 1.0000000001
        self.assertTrue(t.converged(result=q.compute()))

        #test at high
        x.value = -2.0
        x.high = 2.0
        self.assertFalse(t.converged(result=q.compute()))
        x.value = 0.0
        x.high = 0.0
        self.assertTrue(t.converged(result=q.compute()))

        #test at low
        x.high = None
        x.value = 0.0
        x.low = 0.0
        self.assertFalse(t.converged(result=q.compute()))
        x.value = 2.0
        x.low = 2.0
        self.assertTrue(t.converged(result=q.compute()))

class test_ValueTest(unittest.TestCase):
    """Unit tests for relentless.optimize.ValueTest"""

    def test_init(self):
        """Test creation with data."""
        x = relentless.variable.DesignVariable(value=3.0)

        #test scalar tolerance
        t = relentless.optimize.ValueTest(absolute=1e-8, relative=1e-5, value=3.0)
        self.assertAlmostEqual(t.absolute, 1e-8)
        self.assertAlmostEqual(t.relative, 1e-5)
        self.assertAlmostEqual(t.value, 3.0)

        #test dictionary of parameters
        t.absolute = {x:1e-8}
        t.relative = {x:1e-5}
        t.value = {x:3.0}
        self.assertDictEqual(t.absolute, {x:1e-8})
        self.assertDictEqual(t.relative, {x:1e-5})
        self.assertDictEqual(t.value, {x:3.0})

    def test_converged(self):
        """Test converged method."""
        x = relentless.variable.DesignVariable(value=3.0)
        q = QuadraticObjective(x=x)

        #test scalar tolerance
        t = relentless.optimize.ValueTest(absolute=0.2, relative=0.2, value=2.0)
        self.assertFalse(t.converged(result=q.compute()))
        x.value = 2.4
        self.assertTrue(t.converged(result=q.compute()))

        #test dictionary of parameters
        t.absolute = {x:1e-8}
        t.relative = {x:1e-5}
        t.value = {x:2.5}
        self.assertFalse(t.converged(result=q.compute()))
        x.value = 2.5000000001
        self.assertTrue(t.converged(result=q.compute()))

class AnyTest(unittest.TestCase):
    """Unit tests for relentless.optimize.AnyTest"""

    def test_init(self):
        """Test creation with data."""
        x = relentless.variable.DesignVariable(value=3.0)
        t1 = relentless.optimize.GradientTest(tolerance=1e-8)
        t2 = relentless.optimize.ValueTest(absolute=1e-5, relative=1e-3, value=2.0)
        t3 = relentless.optimize.ValueTest(absolute=1e-5, relative=1e-3, value=1.0)

        t = relentless.optimize.AnyTest(t1,t2,t3)
        self.assertCountEqual(t.tests, (t1,t2,t3))

    def test_converged(self):
        """Test converged method."""
        x = relentless.variable.DesignVariable(value=3.0)
        q = QuadraticObjective(x=x)
        t1 = relentless.optimize.GradientTest(tolerance=1e-8)
        t2 = relentless.optimize.ValueTest(absolute=1e-5, relative=1e-3, value=2.0)
        t3 = relentless.optimize.ValueTest(absolute=1e-5, relative=1e-3, value=1.0)

        t = relentless.optimize.AnyTest(t1,t2,t3)
        self.assertFalse(t.converged(result=q.compute()))

        x.value = 2.00000001
        self.assertTrue(t.converged(result=q.compute()))

        x.value = 1.00000001
        self.assertTrue(t.converged(result=q.compute()))

class AllTest(unittest.TestCase):
    """Unit tests for relentless.optimize.AllTest"""

    def test_init(self):
        """Test creation with data."""
        x = relentless.variable.DesignVariable(value=3.0)
        t1 = relentless.optimize.GradientTest(tolerance=1e-8)
        t2 = relentless.optimize.ValueTest(absolute=1e-5, relative=1e-3, value=2.0)
        t3 = relentless.optimize.ValueTest(absolute=1e-5, relative=1e-3, value=1.0)

        t = relentless.optimize.AllTest(t1,t2,t3)
        self.assertCountEqual(t.tests, (t1,t2,t3))

    def test_converged(self):
        """Test converged method."""
        x = relentless.variable.DesignVariable(value=3.0)
        q = QuadraticObjective(x=x)
        t1 = relentless.optimize.GradientTest(tolerance=1e-8)
        t2 = relentless.optimize.ValueTest(absolute=1e-5, relative=1e-3, value=2.0)
        t3 = relentless.optimize.ValueTest(absolute=1e-5, relative=1e-3, value=1.0)

        t = relentless.optimize.AllTest(t1,t2,t3)
        self.assertFalse(t.converged(result=q.compute()))

        x.value = 2.0000001
        self.assertFalse(t.converged(result=q.compute()))

        t2.value = 1.0
        x.value = 1.000000001
        self.assertTrue(t.converged(result=q.compute()))

class OrTest(unittest.TestCase):
    """Unit tests for relentless.optimize.OrTest"""

    def test_init(self):
        """Test creation with data."""
        x = relentless.variable.DesignVariable(value=3.0)
        t1 = relentless.optimize.GradientTest(tolerance=1e-8)
        t2 = relentless.optimize.ValueTest(absolute=1e-5, relative=1e-3, value=2.0)

        t = relentless.optimize.OrTest(t1,t2)
        self.assertCountEqual(t.tests, (t1,t2))

    def test_converged(self):
        """Test converged method."""
        x = relentless.variable.DesignVariable(value=3.0)
        q = QuadraticObjective(x=x)
        t1 = relentless.optimize.GradientTest(tolerance=1e-8)
        t2 = relentless.optimize.ValueTest(absolute=1e-5, relative=1e-3, value=2.0)

        t = relentless.optimize.OrTest(t1,t2)
        self.assertFalse(t.converged(result=q.compute()))

        x.value = 2.000000001
        self.assertTrue(t.converged(result=q.compute()))

        x.value = 1.000000001
        self.assertTrue(t.converged(result=q.compute()))

class AndTest(unittest.TestCase):
    """Unit tests for relentless.optimize.AndTest"""

    def test_init(self):
        """Test creation with data."""
        x = relentless.variable.DesignVariable(value=3.0)
        t1 = relentless.optimize.GradientTest(tolerance=1e-8)
        t2 = relentless.optimize.ValueTest(absolute=1e-5, relative=1e-3, value=1.0)

        t = relentless.optimize.AndTest(t1,t2)
        self.assertCountEqual(t.tests, (t1,t2))

    def test_converged(self):
        """Test converged method."""
        x = relentless.variable.DesignVariable(value=3.0)
        q = QuadraticObjective(x=x)
        t1 = relentless.optimize.GradientTest(tolerance=1e-8)
        t2 = relentless.optimize.ValueTest(absolute=1e-5, relative=1e-3, value=2.0)

        t = relentless.optimize.AndTest(t1,t2)
        self.assertFalse(t.converged(result=q.compute()))

        x.value = 2.0000001
        self.assertFalse(t.converged(result=q.compute()))

        t2.value = 1.0
        x.value = 1.000000001
        self.assertTrue(t.converged(result=q.compute()))
