"""Unit tests for core module."""
import unittest

import sys
sys.path.append('../')
import relentless

class test_Interpolator(unittest.TestCase):
    """Unit tests for core.Interpolator."""

    def test_init(self):
        """Test creation from data."""
        raise NotImplementedError()

    def test_call(self):
        """Test calls, both scalar and array."""
        raise NotImplementedError()

    def test_extrap(self):
        """Test extrapolation calls."""
        raise NotImplementedError()

class test_PairMatrix(unittest.TestCase):
    """Unit tests for core.PairMatrix."""

    def test_init(self):
        """Test construction with different list types."""
        raise NotImplementedError()

    def test_accessors(self):
        """Test get and set methods on pairs."""
        raise NotImplementedError()

    def test_iteration(self):
        """Test iteration on the matrix."""
        raise NotImplementedError()

class test_TypeDict(unittest.TestCase):
    """Unit tests for core.TypeDict."""

    def test_init(self):
        """Test construction with different list types."""
        raise NotImplementedError()

    def test_accessors(self):
        """Test get and set methods on types."""
        raise NotImplementedError()

    def test_iteration(self):
        """Test iteration on the dictionary."""
        raise NotImplementedError()

    def test_copy(self):
        """Test copying custom dict to standard dict."""
        raise NotImplementedError()

class test_Variable(unittest.TestCase):
    """Unit tests for core.Variable."""

    def test_init(self):
        """Test construction with different bounds."""
        v = relentless.core.Variable(value=1.0, const=True)
        self.assertAlmostEqual(v.value, 1.0)
        self.assertEqual(v.const, True)

        v = relentless.core.Variable(value=1.2, low=0.0, high=2.0)
        self.assertAlmostEqual(v.value, 1.2)
        self.assertAlmostEqual(v.low, 0.0)
        self.assertAlmostEqual(v.high, 2.0)

        v = relentless.core.Variable(value=-1.2, low=0.5)
        self.assertAlmostEqual(v.value, 0.5)
        self.assertAlmostEqual(v.low, 0.5)

        v = relentless.core.Variable(value=2.2, high=2.0)
        self.assertAlmostEqual(v.value, 2.0)
        self.assertAlmostEqual(v.high, 2.0)

        with self.assertRaises(ValueError):
            v = relentless.core.Variable(value=4)

    def test_clamp(self):
        """Test methods for clamping values with bounds."""
        v = relentless.core.Variable(value=0.0, low=2.0)

        x = v.clamp(1.0)
        self.assertAlmostEqual(x[0], 2.0)
        self.assertEqual(x[1], v.State.LOW)

        x = v.clamp(2.0)
        self.assertAlmostEqual(x[0], 2.0)
        self.assertEqual(x[1], v.State.LOW)

        x = v.clamp(3.0)
        self.assertAlmostEqual(x[0], 3.0)
        self.assertEqual(x[1], v.State.FREE)

        v = relentless.core.Variable(value=0.0, low=0.0, high=2.0)

        x = v.clamp(-1.0)
        self.assertAlmostEqual(x[0], 0.0)
        self.assertEqual(x[1], v.State.LOW)

        x = v.clamp(1.0)
        self.assertAlmostEqual(x[0], 1.0)
        self.assertEqual(x[1], v.State.FREE)

        x = v.clamp(2.5)
        self.assertAlmostEqual(x[0], 2.0)
        self.assertEqual(x[1], v.State.HIGH)

        v = relentless.core.Variable(value=0.0)

        x = v.clamp(1.0)
        self.assertAlmostEqual(x[0], 1.0)
        self.assertEqual(x[1], v.State.FREE)

    def test_value(self):
        """Test methods for setting values and checking bounds."""
        v = relentless.core.Variable(value=0.0, low=-1.0, high=1.0)
        self.assertAlmostEqual(v.value, 0.0)
        self.assertEqual(v.state, v.State.FREE)

        v.value = -1.5
        self.assertAlmostEqual(v.value, -1.0)
        self.assertEqual(v.state, v.State.LOW)

        v.value = 3.0
        self.assertAlmostEqual(v.value, 1.0)
        self.assertEqual(v.state, v.State.HIGH)

        with self.assertRaises(ValueError):
            v.value = 0

if __name__ == '__main__':
    unittest.main()
