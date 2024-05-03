"""Unit tests for criteria module."""

import unittest

import relentless

from .test_objective import QuadraticObjective


class test_Tolerance(unittest.TestCase):
    """Unit tests for relentless.optimize.Tolerance"""

    def test_init(self):
        """Test creation with data."""
        x = relentless.model.IndependentVariable(value=3.0)
        y = relentless.model.IndependentVariable(value=4.0)

        t = relentless.optimize.Tolerance(absolute=1.0, relative=0.5)
        self.assertAlmostEqual(t.absolute[x], 1.0)
        self.assertAlmostEqual(t.relative[x], 0.5)
        self.assertAlmostEqual(t.absolute[y], 1.0)
        self.assertAlmostEqual(t.relative[y], 0.5)

        # test changing tolerances by scalar
        t.absolute.default = 1.1
        t.relative.default = 0.6
        self.assertAlmostEqual(t.absolute[x], 1.1)
        self.assertAlmostEqual(t.relative[x], 0.6)
        self.assertAlmostEqual(t.absolute[y], 1.1)
        self.assertAlmostEqual(t.relative[y], 0.6)

        # test changing tolerances by key
        t.absolute[x] = 1.2
        t.relative[x] = 0.7
        self.assertAlmostEqual(t.absolute[x], 1.2)
        self.assertAlmostEqual(t.relative[x], 0.7)
        self.assertAlmostEqual(t.absolute[y], 1.1)
        self.assertAlmostEqual(t.relative[y], 0.6)

    def test_isclose(self):
        """Test isclose method."""
        x = relentless.model.IndependentVariable(value=3.0)

        t = relentless.optimize.Tolerance(absolute=0.1, relative=0.1)
        self.assertFalse(t.isclose(x.value, 2.5, key=x))
        self.assertFalse(t.isclose(x.value, 2.5))
        self.assertTrue(t.isclose(x.value, 2.7, key=x))
        self.assertTrue(t.isclose(x.value, 2.7))

        # test invalid tolerances
        with self.assertRaises(ValueError):
            t = relentless.optimize.Tolerance(absolute=-0.1, relative=0.1)
            t.isclose(x.value, 2.5)
        with self.assertRaises(ValueError):
            t = relentless.optimize.Tolerance(absolute=0.1, relative=-0.1)
            t.isclose(x.value, 2.5)
        with self.assertRaises(ValueError):
            t = relentless.optimize.Tolerance(absolute=0.1, relative=1.1)
            t.isclose(x.value, 2.5)


class test_GradientTest(unittest.TestCase):
    """Unit tests for relentless.optimize.GradientTest"""

    def test_init(self):
        """Test creation with data."""
        x = relentless.model.IndependentVariable(value=3.0)

        t = relentless.optimize.GradientTest(tolerance=1e-8, variables=x)
        self.assertAlmostEqual(t.tolerance[x], 1e-8)
        self.assertAlmostEqual(t.tolerance.default, 1e-8)

        # change tolerance
        t.tolerance[x] = 1e-5
        self.assertAlmostEqual(t.tolerance[x], 1e-5)
        self.assertAlmostEqual(t.tolerance.default, 1e-8)

    def test_converged(self):
        """Test converged method."""
        x = relentless.model.IndependentVariable(value=3.0)
        q = QuadraticObjective(x=x)

        t = relentless.optimize.GradientTest(tolerance=1e-8, variables=x)
        self.assertFalse(t.converged(result=q.compute(x)))
        x.value = 0.999999999
        self.assertTrue(t.converged(result=q.compute(x)))

        # test at high
        x.value = -2.0
        x.high = 2.0
        self.assertFalse(t.converged(result=q.compute(x)))
        x.value = 0.0
        x.high = 0.0
        self.assertTrue(t.converged(result=q.compute(x)))

        # test at low
        x.high = None
        x.value = 0.0
        x.low = 0.0
        self.assertFalse(t.converged(result=q.compute(x)))
        x.value = 2.0
        x.low = 2.0
        self.assertTrue(t.converged(result=q.compute(x)))


class test_ValueTest(unittest.TestCase):
    """Unit tests for relentless.optimize.ValueTest"""

    def test_init(self):
        """Test creation with data."""
        # test default values
        t = relentless.optimize.ValueTest(value=2.5)
        self.assertAlmostEqual(t.absolute, 1e-8)
        self.assertAlmostEqual(t.relative, 1e-5)
        self.assertAlmostEqual(t.value, 2.5)

        # non-default values
        t = relentless.optimize.ValueTest(absolute=1e-7, relative=1e-4, value=3.0)
        self.assertAlmostEqual(t.absolute, 1e-7)
        self.assertAlmostEqual(t.relative, 1e-4)
        self.assertAlmostEqual(t.value, 3.0)

        # change parameters
        t.absolute = 1e-9
        t.relative = 1e-5
        t.value = 1.5
        self.assertAlmostEqual(t.absolute, 1e-9)
        self.assertAlmostEqual(t.relative, 1e-5)
        self.assertAlmostEqual(t.value, 1.5)

    def test_converged(self):
        """Test converged method."""
        x = relentless.model.IndependentVariable(value=3.0)
        q = QuadraticObjective(x=x)

        t = relentless.optimize.ValueTest(absolute=0.2, relative=0.2, value=1.0)
        self.assertFalse(t.converged(result=q.compute(x)))
        x.value = 1.999999999
        self.assertTrue(t.converged(result=q.compute(x)))

        t = relentless.optimize.ValueTest(value=9.0)
        self.assertFalse(t.converged(result=q.compute(x)))
        x.value = 3.9999999999
        self.assertTrue(t.converged(result=q.compute(x)))


class AnyTest(unittest.TestCase):
    """Unit tests for relentless.optimize.AnyTest"""

    def test_init(self):
        """Test creation with data."""
        x = relentless.model.IndependentVariable(value=3.0)
        t1 = relentless.optimize.GradientTest(tolerance=1e-8, variables=x)
        t2 = relentless.optimize.ValueTest(value=2.0)
        t3 = relentless.optimize.ValueTest(value=1.0)

        t = relentless.optimize.AnyTest(t1, t2, t3)
        self.assertCountEqual(t.tests, (t1, t2, t3))

    def test_converged(self):
        """Test converged method."""
        x = relentless.model.IndependentVariable(value=3.0)
        q = QuadraticObjective(x=x)
        t1 = relentless.optimize.GradientTest(tolerance=1e-8, variables=x)
        t2 = relentless.optimize.ValueTest(value=1.0)
        t3 = relentless.optimize.ValueTest(value=0.0)

        t = relentless.optimize.AnyTest(t1, t2, t3)
        self.assertFalse(t.converged(result=q.compute(x)))

        x.value = 2.00000001
        self.assertTrue(t.converged(result=q.compute(x)))

        x.value = 1.00000001
        self.assertTrue(t.converged(result=q.compute(x)))


class AllTest(unittest.TestCase):
    """Unit tests for relentless.optimize.AllTest"""

    def test_init(self):
        """Test creation with data."""
        x = relentless.model.IndependentVariable(value=3.0)
        t1 = relentless.optimize.GradientTest(tolerance=1e-8, variables=x)
        t2 = relentless.optimize.ValueTest(value=1.0)
        t3 = relentless.optimize.ValueTest(value=0.0)

        t = relentless.optimize.AllTest(t1, t2, t3)
        self.assertCountEqual(t.tests, (t1, t2, t3))

    def test_converged(self):
        """Test converged method."""
        x = relentless.model.IndependentVariable(value=3.0)
        q = QuadraticObjective(x=x)
        t1 = relentless.optimize.GradientTest(tolerance=1e-8, variables=x)
        t2 = relentless.optimize.ValueTest(value=1.0)
        t3 = relentless.optimize.ValueTest(value=0.0)

        t = relentless.optimize.AllTest(t1, t2, t3)
        self.assertFalse(t.converged(result=q.compute(x)))

        x.value = 1.999999999
        self.assertFalse(t.converged(result=q.compute(x)))

        t2.value = 0.0
        x.value = 0.999999999
        self.assertTrue(t.converged(result=q.compute(x)))


class OrTest(unittest.TestCase):
    """Unit tests for relentless.optimize.OrTest"""

    def test_init(self):
        """Test creation with data."""
        x = relentless.model.IndependentVariable(value=3.0)
        t1 = relentless.optimize.GradientTest(tolerance=1e-8, variables=x)
        t2 = relentless.optimize.ValueTest(value=1.0)

        t = relentless.optimize.OrTest(t1, t2)
        self.assertCountEqual(t.tests, (t1, t2))

    def test_converged(self):
        """Test converged method."""
        x = relentless.model.IndependentVariable(value=3.0)
        q = QuadraticObjective(x=x)
        t1 = relentless.optimize.GradientTest(tolerance=1e-8, variables=x)
        t2 = relentless.optimize.ValueTest(value=1.0)

        t = relentless.optimize.OrTest(t1, t2)
        self.assertFalse(t.converged(result=q.compute(x)))

        x.value = 1.9999999999
        self.assertTrue(t.converged(result=q.compute(x)))

        x.value = 0.9999999999
        self.assertTrue(t.converged(result=q.compute(x)))


class AndTest(unittest.TestCase):
    """Unit tests for relentless.optimize.AndTest"""

    def test_init(self):
        """Test creation with data."""
        x = relentless.model.IndependentVariable(value=3.0)
        t1 = relentless.optimize.GradientTest(tolerance=1e-8, variables=x)
        t2 = relentless.optimize.ValueTest(value=1.0)

        t = relentless.optimize.AndTest(t1, t2)
        self.assertCountEqual(t.tests, (t1, t2))

    def test_converged(self):
        """Test converged method."""
        x = relentless.model.IndependentVariable(value=3.0)
        q = QuadraticObjective(x=x)
        t1 = relentless.optimize.GradientTest(tolerance=1e-8, variables=x)
        t2 = relentless.optimize.ValueTest(value=1.0)

        t = relentless.optimize.AndTest(t1, t2)
        self.assertFalse(t.converged(result=q.compute(x)))

        x.value = 1.999999999
        self.assertFalse(t.converged(result=q.compute(x)))

        t2.value = 0.0
        x.value = 0.999999999
        self.assertTrue(t.converged(result=q.compute(x)))


if __name__ == "__main__":
    unittest.main()
