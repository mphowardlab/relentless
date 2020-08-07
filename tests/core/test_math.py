"""Unit tests for core.math module."""
import unittest

import numpy as np

import relentless

class test_Interpolator(unittest.TestCase):
    """Unit tests for relentless.Interpolator."""

    def test_init(self):
        """Test creation from data."""
        #test construction with tuple input
        f = relentless.Interpolator(x=(-1,0,1), y=(-2,0,2))
        self.assertEqual(f.domain, (-1,1))

        #test construction with list input
        f = relentless.Interpolator(x=[-1,0,1], y=[-2,0,2])
        self.assertEqual(f.domain, (-1,1))

        #test construction with numpy array input
        f = relentless.Interpolator(x=np.array([-1,0,1]),
                                         y=np.array([-2,0,2]))
        self.assertEqual(f.domain, (-1,1))

        #test construction with mixed input
        f = relentless.Interpolator(x=[-1,0,1], y=(-2,0,2))
        self.assertEqual(f.domain, (-1,1))

        #test construction with scalar input
        with self.assertRaises(ValueError):
            f = relentless.Interpolator(x=1, y=2)

        #test construction with 2d-array input
        with self.assertRaises(ValueError):
            f = relentless.Interpolator(x=np.array([[-1,0,1], [-2,2,4]]),
                                             y=np.array([[-1,0,1], [-2,2,4]]))

        #test construction with x and y having different lengths
        with self.assertRaises(ValueError):
            f = relentless.Interpolator(x=[-1,0], y=[-2,0,2])

        #test construction with non-strictly-increasing domain
        with self.assertRaises(ValueError):
            f = relentless.Interpolator(x=(0,1,-1), y=(0,2,-2))

    def test_call(self):
        """Test calls, both scalar and array."""
        f = relentless.Interpolator(x=(-1,0,1), y=(-2,0,2))

        #test scalar call
        self.assertAlmostEqual(f(-0.5), -1.0)
        self.assertAlmostEqual(f(0.5), 1.0)

        #test array call
        np.testing.assert_allclose(f([-0.5,0.5]), [-1.0,1.0])

    def test_derivative(self):
        """Test derivative function, both scalar and array."""
        f = relentless.Interpolator(x=(-2,-1,0,1,2), y=(4,1,0,1,4))

        #test scalar call
        d = f.derivative(x=1.5, n=1)
        self.assertAlmostEqual(d, 3.0)

        #test array call
        d = f.derivative(x=np.array([-3.0,-0.5,0.5,3.0]), n=1)
        np.testing.assert_allclose(d, np.array([0.0,-1.0,1.0,0.0]))

    def test_extrap(self):
        """Test extrapolation calls."""
        f = relentless.Interpolator(x=(-1,0,1), y=(-2,0,2))

        #test extrap below lo
        self.assertAlmostEqual(f(-2), -2.0)

        #test extrap above hi
        self.assertAlmostEqual(f(2), 2.0)

        #test extrap below low and above hi
        np.testing.assert_allclose(f([-2,2]), [-2.0,2.0])

        #test combined extrapolation and interpolation
        np.testing.assert_allclose(f([-2,0.5,2]), [-2.0,1.0,2.0])

if __name__ == '__main__':
    unittest.main()
