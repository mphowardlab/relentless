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

class test_Spline(unittest.TestCase):
    """Unit tests for pair.Spline"""

    def test_init(self):
        """Test creation from data"""
        #test diff mode
        s = pair.Spline(types=('1',), num_knots=3)
        self.assertEqual(s.num_knots, 3)
        self.assertEqual(s.mode, 'diff')
        coeff = potential.CoefficientMatrix(types=('1',),
                                            params=('r-0','r-1','r-2','knot-0','knot-1','knot-2','rmin','rmax','shift'),
                                            default={'rmin':False,'rmax':False,'shift':False})
        self.assertCountEqual(s.coeff.types, coeff.types)
        self.assertCountEqual(s.coeff.params, coeff.params)
        self.assertDictEqual(s.coeff.default.todict(), coeff.default.todict())

        #test value mode
        s = pair.Spline(types=('1',), num_knots=3, mode='value')
        self.assertEqual(s.num_knots, 3)
        self.assertEqual(s.mode, 'value')
        coeff = potential.CoefficientMatrix(types=('1',),
                                            params=('r-0','r-1','r-2','knot-0','knot-1','knot-2','rmin','rmax','shift'),
                                            default={'rmin':False,'rmax':False,'shift':False})
        self.assertCountEqual(s.coeff.types, coeff.types)
        self.assertCountEqual(s.coeff.params, coeff.params)
        self.assertDictEqual(s.coeff.default.todict(), coeff.default.todict())


        #test invalid number of knots
        with self.assertRaises(ValueError):
            s = pair.Spline(types=('1',), num_knots=1)

        #test invalid mode
        with self.assertRaises(ValueError):
            s = pair.Spline(types=('1',), num_knots=3, mode='val')

    def test_from_array(self):
        """Test from_array method"""
        r_arr = [1,2,3]
        u_arr = [9,4,1]

        #test diff mode
        s = pair.Spline(types=('1',), num_knots=3)
        s.from_array(pair=('1','1'), r=r_arr, u=u_arr)

        self.assertEqual(s.coeff['1','1']['r-0'].value, 1)
        self.assertEqual(s.coeff['1','1']['r-0'].const, True)
        self.assertEqual(s.coeff['1','1']['r-1'].value, 2)
        self.assertEqual(s.coeff['1','1']['r-1'].const, True)
        self.assertEqual(s.coeff['1','1']['r-2'].value, 3)
        self.assertEqual(s.coeff['1','1']['r-2'].const, True)

        self.assertEqual(s.coeff['1','1']['knot-0'].value, 5)
        self.assertEqual(s.coeff['1','1']['knot-0'].const, False)
        self.assertEqual(s.coeff['1','1']['knot-1'].value, 3)
        self.assertEqual(s.coeff['1','1']['knot-1'].const, False)
        self.assertEqual(s.coeff['1','1']['knot-2'].value, 1)
        self.assertEqual(s.coeff['1','1']['knot-2'].const, True)

        #test value mode
        s = pair.Spline(types=('1',), num_knots=3, mode='value')
        s.from_array(pair=('1','1'), r=r_arr, u=u_arr)

        self.assertEqual(s.coeff['1','1']['r-0'].value, 1)
        self.assertEqual(s.coeff['1','1']['r-0'].const, True)
        self.assertEqual(s.coeff['1','1']['r-1'].value, 2)
        self.assertEqual(s.coeff['1','1']['r-1'].const, True)
        self.assertEqual(s.coeff['1','1']['r-2'].value, 3)
        self.assertEqual(s.coeff['1','1']['r-2'].const, True)

        self.assertEqual(s.coeff['1','1']['knot-0'].value, 9)
        self.assertEqual(s.coeff['1','1']['knot-0'].const, False)
        self.assertEqual(s.coeff['1','1']['knot-1'].value, 4)
        self.assertEqual(s.coeff['1','1']['knot-1'].const, False)
        self.assertEqual(s.coeff['1','1']['knot-2'].value, 1)
        self.assertEqual(s.coeff['1','1']['knot-2'].const, True)

        #test invalid r and u shapes
        r_arr = [2,3]
        with self.assertRaises(ValueError):
            s.from_array(pair=('1','1'), r=r_arr, u=u_arr)

        r_arr = [1,2,3]
        u_arr = [1,2]
        with self.assertRaises(ValueError):
            s.from_array(pair=('1','1'), r=r_arr, u=u_arr)

    def test_energy(self):
        """Test _energy method"""
        r_arr = [1,2,3]
        u_arr = [9,4,1]

        #test diff mode
        s = pair.Spline(types=('1',), num_knots=3)
        s.from_array(pair=('1','1'), r=r_arr, u=u_arr)
        u_actual = np.array([6.25,2.25,1])
        u = s._energy(pair=('1','1'), r=[1.5,2.5,3.5])
        np.testing.assert_allclose(u, u_actual)

        #test value mode
        s = pair.Spline(types=('1',), num_knots=3, mode='value')
        s.from_array(pair=('1','1'), r=r_arr, u=u_arr)
        u_actual = np.array([6.25,2.25,1])
        u = s._energy(pair=('1','1'), r=[1.5,2.5,3.5])
        np.testing.assert_allclose(u, u_actual)

    def test_force(self):
        """Test _force method"""
        r_arr = [1,2,3]
        u_arr = [9,4,1]

        #test diff mode
        s = pair.Spline(types=('1',), num_knots=3)
        s.from_array(pair=('1','1'), r=r_arr, u=u_arr)
        f_actual = np.array([5,3,0])
        f = s._force(pair=('1','1'), r=[1.5,2.5,3.5])
        np.testing.assert_allclose(f, f_actual)

        #test value mode
        s = pair.Spline(types=('1',), num_knots=3, mode='value')
        s.from_array(pair=('1','1'), r=r_arr, u=u_arr)
        f_actual = np.array([5,3,0])
        f = s._force(pair=('1','1'), r=[1.5,2.5,3.5])
        np.testing.assert_allclose(f, f_actual)

    def test_derivative(self):
        """Test _derivative method"""
        r_arr = [1,2,3]
        u_arr = [9,4,1]

        #test diff mode
        s = pair.Spline(types=('1',), num_knots=3)
        s.from_array(pair=('1','1'), r=r_arr, u=u_arr)
        d_actual = np.array([1.125,0.625,0])
        d = s._derivative(pair=('1','1'), param='knot-1', r=[1.5,2.5,3.5])
        np.testing.assert_allclose(d, d_actual)

        #test value mode
        s = pair.Spline(types=('1',), num_knots=3, mode='value')
        s.from_array(pair=('1','1'), r=r_arr, u=u_arr)
        d_actual = np.array([0.75,0.75,0])
        d = s._derivative(pair=('1','1'), param='knot-1', r=[1.5,2.5,3.5])
        np.testing.assert_allclose(d, d_actual)

class test_Yukawa(unittest.TestCase):
    """Unit tests for pair.Yukawa"""

    def test_init(self):
        """Test creation from data"""
        y = pair.Yukawa(types=('1','2'))
        coeff = potential.CoefficientMatrix(types=('1','2'), params=('epsilon','kappa','rmin','rmax','shift'),
                                            default={'rmin':False,'rmax':False,'shift':False})
        self.assertCountEqual(y.coeff.types, coeff.types)
        self.assertCountEqual(y.coeff.params, coeff.params)
        self.assertDictEqual(y.coeff.default.todict(), coeff.default.todict())

    def test_energy(self):
        """Test _energy method"""
        y = pair.Yukawa(types=('1',))

        #test scalar r
        r_input = 0.5
        u_actual = 1.5576016
        u = y._energy(r=r_input, epsilon=1.0, kappa=0.5)
        self.assertAlmostEqual(u, u_actual)

        #test array r
        r_input = np.array([0,1,1.5])
        u_actual = np.array([np.inf,0.60653066,0.31491104])
        u = y._energy(r=r_input, epsilon=1.0, kappa=0.5)
        np.testing.assert_allclose(u, u_actual)

        #test negative kappa
        with self.assertRaises(ValueError):
            u = y._energy(r=r_input, epsilon=1.0, kappa=-1.0)

    def test_force(self):
        """Test _force method"""
        y = pair.Yukawa(types=('1',))

        #test scalar r
        r_input = 0.5
        f_actual = 3.8940039
        f = y._force(r=r_input, epsilon=1.0, kappa=0.5)
        self.assertAlmostEqual(f, f_actual)

        #test array r
        r_input = np.array([0,1,1.5])
        f_actual = np.array([np.inf,0.90979599,0.36739621])
        f = y._force(r=r_input, epsilon=1.0, kappa=0.5)
        np.testing.assert_allclose(f, f_actual)

        #test negative kappa
        with self.assertRaises(ValueError):
            u = y._force(r=r_input, epsilon=1.0, kappa=-1.0)

    def test_derivative(self):
        """Test _derivative method"""
        y = pair.Yukawa(types=('1',))

        #w.r.t. epsilon
        #test scalar r
        r_input = 0.5
        d_actual = 1.5576016
        d = y._derivative(param='epsilon', r=r_input, epsilon=1.0, kappa=0.5)
        self.assertAlmostEqual(d, d_actual)

        #test array r
        r_input = np.array([0,1,1.5])
        d_actual = np.array([np.inf,0.60653066,0.31491104])
        d = y._derivative(param='epsilon', r=r_input, epsilon=1.0, kappa=0.5)
        np.testing.assert_allclose(d, d_actual)

        #w.r.t. kappa
        #test scalar r
        r_input = 0.5
        d_actual = -0.77880078
        d = y._derivative(param='kappa', r=r_input, epsilon=1.0, kappa=0.5)
        self.assertAlmostEqual(d, d_actual)

        #test array r
        r_input = np.array([0,1,1.5])
        d_actual = np.array([-1,-0.60653066,-0.47236655])
        d = y._derivative(param='kappa', r=r_input, epsilon=1.0, kappa=0.5)
        np.testing.assert_allclose(d, d_actual)

        #test negative kappa
        with self.assertRaises(ValueError):
            u = y._derivative(param='kappa', r=r_input, epsilon=1.0, kappa=-1.0)

        #test invalid param
        with self.assertRaises(ValueError):
            u = y._derivative(param='kapppa', r=r_input, epsilon=1.0, kappa=1.0)

if __name__ == '__main__':
    unittest.main()
