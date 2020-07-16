"""Unit tests for pair module."""
import unittest

import numpy as np

from relentless import core
from relentless.potential import pair
from relentless.potential import potential

class test_LennardJones(unittest.TestCase):
    """Unit tests for pair.LennardJones"""

    def test_init(self):
        """Test creation from data"""
        lj = pair.LennardJones(types=('1','2'))
        coeff = potential.CoefficientMatrix(types=('1','2'), params=('epsilon','sigma','rmin','rmax','shift'),
                                            default={'rmin':False,'rmax':False,'shift':False})
        self.assertCountEqual(lj.coeff.types, coeff.types)
        self.assertCountEqual(lj.coeff.params, coeff.params)
        self.assertDictEqual(lj.coeff.default.todict(), coeff.default.todict())

    def test_energy(self):
        """Test _energy method"""
        lj = pair.LennardJones(types=('1',))

        #test scalar r
        r_input = 0.5
        u_actual = 0
        u = lj._energy(r=r_input, epsilon=1.0, sigma=0.5)
        self.assertAlmostEqual(u, u_actual)

        #test array r
        r_input = np.array([0,1,1.5])
        u_actual = np.array([np.inf,-0.061523438,-0.0054794417])
        u = lj._energy(r=r_input, epsilon=1.0, sigma=0.5)
        np.testing.assert_allclose(u, u_actual)

        #test negative sigma
        with self.assertRaises(ValueError):
            u = lj._energy(r=r_input, epsilon=1.0, sigma=-1.0)

    def test_force(self):
        """Test _force method"""
        lj = pair.LennardJones(types=('1',))

        #test scalar r
        r_input = 0.5
        f_actual = 48
        f = lj._force(r=r_input, epsilon=1.0, sigma=0.5)
        self.assertAlmostEqual(f, f_actual)

        #test array r
        r_input = np.array([0,1,1.5])
        f_actual = np.array([np.inf,-0.36328125,-0.02188766])
        f = lj._force(r=r_input, epsilon=1.0, sigma=0.5)
        np.testing.assert_allclose(f, f_actual)

        #test negative sigma
        with self.assertRaises(ValueError):
            u = lj._force(r=r_input, epsilon=1.0, sigma=-1.0)

    def test_derivative(self):
        """Test _derivative method"""
        lj = pair.LennardJones(types=('1',))

        #w.r.t. epsilon
        #test scalar r
        r_input = 0.5
        d_actual = 0
        d = lj._derivative(param='epsilon', r=r_input, epsilon=1.0, sigma=0.5)
        self.assertAlmostEqual(d, d_actual)

        #test array r
        r_input = np.array([0,1,1.5])
        d_actual = np.array([np.inf,-0.061523438,-0.0054794417])
        d = lj._derivative(param='epsilon', r=r_input, epsilon=1.0, sigma=0.5)
        np.testing.assert_allclose(d, d_actual)

        #w.r.t. sigma
        #test scalar r
        r_input = 0.5
        d_actual = 48
        d = lj._derivative(param='sigma', r=r_input, epsilon=1.0, sigma=0.5)
        self.assertAlmostEqual(d, d_actual)

        #test array r
        r_input = np.array([0,1,1.5])
        d_actual = np.array([np.inf,-0.7265625,-0.06566298])
        d = lj._derivative(param='sigma', r=r_input, epsilon=1.0, sigma=0.5)
        np.testing.assert_allclose(d, d_actual)

        #test negative sigma
        with self.assertRaises(ValueError):
            u = lj._derivative(param='sigma', r=r_input, epsilon=1.0, sigma=-1.0)

        #test invalid param
        with self.assertRaises(ValueError):
            u = lj._derivative(param='simga', r=r_input, epsilon=1.0, sigma=1.0)

if __name__ == '__main__':
    unittest.main()
