"""Unit tests for core.math module."""

import unittest

import numpy

import relentless


class test_AkimaSpline(unittest.TestCase):
    """Unit tests for relentless.math.AkimaSpline."""

    def test_init(self):
        """Test creation from data."""
        # test construction with tuple input
        f = relentless.math.AkimaSpline(x=(-1, 0, 1), y=(-2, 0, 2))
        self.assertEqual(f.domain, (-1, 1))

        # test construction with list input
        f = relentless.math.AkimaSpline(x=[-1, 0, 1], y=[-2, 0, 2])
        self.assertEqual(f.domain, (-1, 1))

        # test construction with numpy array input
        f = relentless.math.AkimaSpline(
            x=numpy.array([-1, 0, 1]), y=numpy.array([-2, 0, 2])
        )
        self.assertEqual(f.domain, (-1, 1))

        # test construction with mixed input
        f = relentless.math.AkimaSpline(x=[-1, 0, 1], y=(-2, 0, 2))
        self.assertEqual(f.domain, (-1, 1))

        # test construction with scalar input
        with self.assertRaises(ValueError):
            f = relentless.math.AkimaSpline(x=1, y=2)

        # test construction with 2d-array input
        with self.assertRaises(ValueError):
            f = relentless.math.AkimaSpline(
                x=numpy.array([[-1, 0, 1], [-2, 2, 4]]),
                y=numpy.array([[-1, 0, 1], [-2, 2, 4]]),
            )

        # test construction with x and y having different lengths
        with self.assertRaises(ValueError):
            f = relentless.math.AkimaSpline(x=[-1, 0], y=[-2, 0, 2])

        # test construction with non-strictly-increasing domain
        with self.assertRaises(ValueError):
            f = relentless.math.AkimaSpline(x=(0, 1, -1), y=(0, 2, -2))

    def test_call(self):
        """Test calls, both scalar and array."""
        f = relentless.math.AkimaSpline(x=(-1, 0, 1), y=(-2, 0, 2))

        # test scalar call
        self.assertAlmostEqual(f(-0.5), -1.0)
        self.assertAlmostEqual(f(0.5), 1.0)

        # test array call
        numpy.testing.assert_allclose(f([-0.5, 0.5]), [-1.0, 1.0])

    def test_derivative(self):
        """Test derivative function, both scalar and array."""
        f = relentless.math.AkimaSpline(x=(-2, -1, 0, 1, 2), y=(4, 1, 0, 1, 4))

        # test scalar call
        d = f.derivative(x=1.5, n=1)
        self.assertAlmostEqual(d, 3.0)

        # test array call
        d = f.derivative(x=numpy.array([-3.0, -0.5, 0.5, 3.0]), n=1)
        numpy.testing.assert_allclose(d, numpy.array([0.0, -1.0, 1.0, 0.0]))

    def test_extrap(self):
        """Test extrapolation calls."""
        f = relentless.math.AkimaSpline(x=(-1, 0, 1), y=(-2, 0, 2))

        # test extrap below lo
        self.assertAlmostEqual(f(-2), -2.0)

        # test extrap above hi
        self.assertAlmostEqual(f(2), 2.0)

        # test extrap below low and above hi
        numpy.testing.assert_allclose(f([-2, 2]), [-2.0, 2.0])

        # test combined extrapolation and interpolation
        numpy.testing.assert_allclose(f([-2, 0.5, 2]), [-2.0, 1.0, 2.0])


class test_KeyedArray(unittest.TestCase):
    """Unit tests for relentless.math.KeyedArray."""

    def test_init(self):
        """Test construction with data."""
        k = relentless.math.KeyedArray(keys=("A", "B"))
        self.assertEqual(dict(k), {"A": None, "B": None})

        k = relentless.math.KeyedArray(keys=("A", "B"), default=2.0)
        self.assertEqual(dict(k), {"A": 2.0, "B": 2.0})

        # invalid key
        with self.assertRaises(KeyError):
            k["C"]

    def test_arithmetic_ops(self):
        """Test arithmetic operations."""
        k1 = relentless.math.KeyedArray(keys=("A", "B"))
        k1.update({"A": 1.0, "B": 2.0})
        k2 = relentless.math.KeyedArray(keys=("A", "B"))
        k2.update({"A": 2.0, "B": 3.0})
        k3 = relentless.math.KeyedArray(keys=("A", "B", "C"))
        k3.update({"A": 3.0, "B": 4.0, "C": 5.0})

        # addition
        k4 = k1 + k2
        self.assertEqual(dict(k4), {"A": 3.0, "B": 5.0})
        k4 += k2
        self.assertEqual(dict(k4), {"A": 5.0, "B": 8.0})
        k4 = k1 + 1
        self.assertEqual(dict(k4), {"A": 2.0, "B": 3.0})
        k4 = 2 + k1
        self.assertEqual(dict(k4), {"A": 3.0, "B": 4.0})
        k4 += 1
        self.assertEqual(dict(k4), {"A": 4.0, "B": 5.0})
        with self.assertRaises(KeyError):
            k4 = k1 + k3
        with self.assertRaises(KeyError):
            k3 += k2

        # subtraction
        k4 = k1 - k2
        self.assertEqual(dict(k4), {"A": -1.0, "B": -1.0})
        k4 -= k2
        self.assertEqual(dict(k4), {"A": -3.0, "B": -4.0})
        k4 = k1 - 1
        self.assertEqual(dict(k4), {"A": 0.0, "B": 1.0})
        k4 = 2 - k1
        self.assertEqual(dict(k4), {"A": 1.0, "B": 0.0})
        k4 -= 1
        self.assertEqual(dict(k4), {"A": 0.0, "B": -1.0})
        with self.assertRaises(KeyError):
            k4 = k1 - k3
        with self.assertRaises(KeyError):
            k3 -= k2

        # multiplication
        k4 = k1 * k2
        self.assertEqual(dict(k4), {"A": 2.0, "B": 6.0})
        k4 *= k2
        self.assertEqual(dict(k4), {"A": 4.0, "B": 18.0})
        k4 = 3 * k1
        self.assertEqual(dict(k4), {"A": 3.0, "B": 6.0})
        k4 = k2 * 3
        self.assertEqual(dict(k4), {"A": 6.0, "B": 9.0})
        k4 *= 3
        self.assertEqual(dict(k4), {"A": 18.0, "B": 27.0})
        with self.assertRaises(KeyError):
            k4 = k1 * k3

        # division
        k4 = k1 / k2
        self.assertEqual(dict(k4), {"A": 0.5, "B": 0.6666666666666666})
        k4 /= k2
        self.assertEqual(dict(k4), {"A": 0.25, "B": 0.2222222222222222})
        k4 = 2 / k2
        self.assertEqual(dict(k4), {"A": 1.0, "B": 0.6666666666666666})
        k4 = k2 / 2
        self.assertEqual(dict(k4), {"A": 1.0, "B": 1.5})
        k4 /= 2
        self.assertEqual(dict(k4), {"A": 0.5, "B": 0.75})
        with self.assertRaises(KeyError):
            k4 = k1 / k3

        # exponentiation
        k4 = k1**k2
        self.assertEqual(dict(k4), {"A": 1.0, "B": 8.0})
        k4 = k2**2
        self.assertEqual(dict(k4), {"A": 4.0, "B": 9.0})

        # negation
        k4 = -k1
        self.assertEqual(dict(k4), {"A": -1.0, "B": -2.0})

    def test_vector_ops(self):
        """Test vector operations."""
        k1 = relentless.math.KeyedArray(keys=("A", "B"))
        k1.update({"A": 1.0, "B": 2.0})
        k2 = relentless.math.KeyedArray(keys=("A", "B"))
        k2.update({"A": 2.0, "B": 3.0})
        k3 = relentless.math.KeyedArray(keys=("A", "B", "C"))
        k3.update({"A": 3.0, "B": 4.0, "C": 5.0})

        # norm
        self.assertAlmostEqual(k1.norm(), numpy.sqrt(5))
        self.assertAlmostEqual(k3.norm(), numpy.sqrt(50))

        # dot product
        self.assertAlmostEqual(k1.dot(k2), 8.0)
        self.assertAlmostEqual(k1.dot(k2), k2.dot(k1))
        with self.assertRaises(KeyError):
            k2.dot(k3)


if __name__ == "__main__":
    unittest.main()
