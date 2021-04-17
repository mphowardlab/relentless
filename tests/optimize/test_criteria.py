"""Unit tests for criteria module."""
import unittest

import relentless

from .test_objective import QuadraticObjective

class test_AbsoluteGradientTest(unittest.TestCase):
    """Unit tests for relentless.optimize.AbsoluteGradientTest"""

    def test_init(self):
        """Test creation with data."""
        x = relentless.variable.DesignVariable(value=3.0)

        #test scalar tolerance
        t = relentless.optimize.AbsoluteGradientTest(tolerance=1e-8)
        self.assertAlmostEqual(t.tolerance, 1e-8)

        #test dictionary of tolerances
        t.tolerance = {x:1e-8}
        self.assertDictEqual(t.tolerance, {x:1e-8})

        #test invalid tolerance
        with self.assertRaises(ValueError):
            t.tolerance = -1e-8
        with self.assertRaises(ValueError):
            t.tolerance = {x:-1e-8}

    def test_converged(self):
        """Test converged method."""
        x = relentless.variable.DesignVariable(value=3.0)
        q = QuadraticObjective(x=x)

        #test scalar tolerance
        t = relentless.optimize.AbsoluteGradientTest(tolerance=1e-8)
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

class test_RelativeGradientTest(unittest.TestCase):
    """Unit tests for relentless.optimize.RelativeGradientTest"""

    def test_init(self):
        """Test creation with data."""
        x = relentless.variable.DesignVariable(value=3.0)

        #test scalar tolerance
        t = relentless.optimize.RelativeGradientTest(tolerance=1e-8)
        self.assertAlmostEqual(t.tolerance, 1e-8)

        #test dictionary of tolerances
        t.tolerance = {x:1e-8}
        self.assertDictEqual(t.tolerance, {x:1e-8})

        #test invalid tolerance
        with self.assertRaises(ValueError):
            t.tolerance = -1e-8
        with self.assertRaises(ValueError):
            t.tolerance = 1.1
        with self.assertRaises(ValueError):
            t.tolerance = {x:-1e-8}
        with self.assertRaises(ValueError):
            t.tolerance = {x:1.1}

    def test_converged(self):
        """Test converged method."""
        x = relentless.variable.DesignVariable(value=3.0)
        q = QuadraticObjective(x=x)

        #test scalar tolerance
        t = relentless.optimize.RelativeGradientTest(tolerance=1e-8)
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
        t = relentless.optimize.ValueTest(tolerance=1e-8)
        self.assertAlmostEqual(t.tolerance, 1e-8)

        #test dictionary of tolerances
        t.tolerance = {x:1e-8}
        self.assertDictEqual(t.tolerance, {x:1e-8})

        #test invalid tolerance
        with self.assertRaises(ValueError):
            t.tolerance = -1e-8
        with self.assertRaises(ValueError):
            t.tolerance = {x:-1e-8}

    def test_converged(self):
        """Test converged method."""
        x = relentless.variable.DesignVariable(value=3.0)
        q = QuadraticObjective(x=x)

        #test scalar tolerance
        t = relentless.optimize.ValueTest(tolerance=1.15)
        self.assertFalse(t.converged(result=q.compute()))
        x.value = 1.1
        self.assertTrue(t.converged(result=q.compute()))

        #test dictionary of tolerances
        x.value = 3.0
        t.tolerance = {x:1.1}
        self.assertFalse(t.converged(result=q.compute()))
        x.value = -100                                      #correct behavior?
        self.assertTrue(t.converged(result=q.compute()))

class AnyTest(unittest.TestCase):
    """Unit tests for relentless.optimize.AnyTest"""

    def test_init(self):
        """Test creation with data."""
        x = relentless.variable.DesignVariable(value=3.0)
        t1 = relentless.optimize.AbsoluteGradientTest(tolerance=1e-8)
        t2 = relentless.optimize.RelativeGradientTest(tolerance={x:1e-9})
        t3 = relentless.optimize.ValueTest(tolerance=1.25)

        t = relentless.optimize.AnyTest(t1,t2,t3)
        self.assertCountEqual(t.tests, (t1,t2,t3))

    def test_converged(self):
        """Test converged method."""
        x = relentless.variable.DesignVariable(value=3.0)
        q = QuadraticObjective(x=x)
        t1 = relentless.optimize.AbsoluteGradientTest(tolerance=1e-8)
        t2 = relentless.optimize.RelativeGradientTest(tolerance={x:1e-9})
        t3 = relentless.optimize.ValueTest(tolerance=1.25)

        t = relentless.optimize.AnyTest(t1,t2,t3)
        self.assertFalse(t.converged(result=q.compute()))

        x.value = 1.2
        self.assertTrue(t.converged(result=q.compute()))

        x.value = 1.0000000001
        self.assertTrue(t.converged(result=q.compute()))

class AllTest(unittest.TestCase):
    """Unit tests for relentless.optimize.AllTest"""

    def test_init(self):
        """Test creation with data."""
        x = relentless.variable.DesignVariable(value=3.0)
        t1 = relentless.optimize.AbsoluteGradientTest(tolerance=1e-8)
        t2 = relentless.optimize.RelativeGradientTest(tolerance={x:1e-9})
        t3 = relentless.optimize.ValueTest(tolerance=1.25)

        t = relentless.optimize.AnyTest(t1,t2,t3)
        self.assertCountEqual(t.tests, (t1,t2,t3))

    def test_converged(self):
        """Test converged method."""
        x = relentless.variable.DesignVariable(value=3.0)
        q = QuadraticObjective(x=x)
        t1 = relentless.optimize.AbsoluteGradientTest(tolerance=1e-8)
        t2 = relentless.optimize.RelativeGradientTest(tolerance={x:1e-9})
        t3 = relentless.optimize.ValueTest(tolerance=1.25)

        t = relentless.optimize.AllTest(t1,t2,t3)
        self.assertFalse(t.converged(result=q.compute()))

        x.value = 1.2
        self.assertFalse(t.converged(result=q.compute()))

        x.value = 1.0000000001
        self.assertTrue(t.converged(result=q.compute()))
