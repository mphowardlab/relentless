"""Unit tests for pair module."""
import unittest

import numpy as np

import relentless

class test_LennardJones(unittest.TestCase):
    """Unit tests for relentless.potential.pair.LennardJones"""

    def test_init(self):
        """Test creation from data"""
        lj = relentless.potential.pair.LennardJones(types=('1',))
        coeff = relentless.potential.potential.PairParameters(types=('1',),
                                                              params=('epsilon','sigma','rmin','rmax','shift'))
        for pair in coeff.pairs:
            coeff[pair]['rmin'] = False
            coeff[pair]['rmax'] = False
            coeff[pair]['shift'] = False
        self.assertCountEqual(lj.coeff.types, coeff.types)
        self.assertCountEqual(lj.coeff.params, coeff.params)

    def test_energy(self):
        """Test _energy method"""
        lj = relentless.potential.pair.LennardJones(types=('1',))

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
        lj = relentless.potential.pair.LennardJones(types=('1',))

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
        lj = relentless.potential.pair.LennardJones(types=('1',))

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
    """Unit tests for relentless.potential.pair.Spline"""

    def test_init(self):
        """Test creation from data"""
        #test diff mode
        s = relentless.potential.pair.Spline(types=('1',), num_knots=3)
        self.assertEqual(s.num_knots, 3)
        self.assertEqual(s.mode, 'diff')
        coeff = relentless.potential.potential.PairParameters(types=('1',),
                                                              params=('r-0','r-1','r-2','knot-0','knot-1','knot-2','rmin','rmax','shift'))
        self.assertCountEqual(s.coeff.types, coeff.types)
        self.assertCountEqual(s.coeff.params, coeff.params)

        #test value mode
        s = relentless.potential.pair.Spline(types=('1',), num_knots=3, mode='value')
        self.assertEqual(s.num_knots, 3)
        self.assertEqual(s.mode, 'value')
        coeff = relentless.potential.potential.PairParameters(types=('1',),
                                                              params=('r-0','r-1','r-2','knot-0','knot-1','knot-2','rmin','rmax','shift'))
        self.assertCountEqual(s.coeff.types, coeff.types)
        self.assertCountEqual(s.coeff.params, coeff.params)

        #test invalid number of knots
        with self.assertRaises(ValueError):
            s = relentless.potential.pair.Spline(types=('1',), num_knots=1)

        #test invalid mode
        with self.assertRaises(ValueError):
            s = relentless.potential.pair.Spline(types=('1',), num_knots=3, mode='val')

    def test_from_array(self):
        """Test from_array method and knots generator"""
        r_arr = [1,2,3]
        u_arr = [9,4,1]
        u_arr_diff = [5,3,1]

        #test diff mode
        s = relentless.potential.pair.Spline(types=('1',), num_knots=3)
        s.from_array(pair=('1','1'), r=r_arr, u=u_arr)

        for i,(r,k) in enumerate(s.knots(pair=('1','1'))):
            self.assertAlmostEqual(r.value, r_arr[i])
            self.assertAlmostEqual(k.value, u_arr_diff[i])
            self.assertEqual(r.const, True)
            if i == s.num_knots-1:
                self.assertEqual(k.const, True)
            else:
                self.assertEqual(k.const, False)

        #test value mode
        s = relentless.potential.pair.Spline(types=('1',), num_knots=3, mode='value')
        s.from_array(pair=('1','1'), r=r_arr, u=u_arr)

        for i,(r,k) in enumerate(s.knots(pair=('1','1'))):
            self.assertAlmostEqual(r.value, r_arr[i])
            self.assertAlmostEqual(k.value, u_arr[i])
            self.assertEqual(r.const, True)
            if i == s.num_knots-1:
                self.assertEqual(k.const, True)
            else:
                self.assertEqual(k.const, False)

        #test invalid r and u shapes
        r_arr = [2,3]
        with self.assertRaises(ValueError):
            s.from_array(pair=('1','1'), r=r_arr, u=u_arr)

        r_arr = [1,2,3]
        u_arr = [1,2]
        with self.assertRaises(ValueError):
            s.from_array(pair=('1','1'), r=r_arr, u=u_arr)

    def test_energy(self):
        """Test energy method"""
        r_arr = [1,2,3]
        u_arr = [9,4,1]

        #test diff mode
        s = relentless.potential.pair.Spline(types=('1',), num_knots=3)
        s.from_array(pair=('1','1'), r=r_arr, u=u_arr)
        u_actual = np.array([6.25,2.25,1])
        u = s.energy(pair=('1','1'), r=[1.5,2.5,3.5])
        np.testing.assert_allclose(u, u_actual)

        #test value mode
        s = relentless.potential.pair.Spline(types=('1',), num_knots=3, mode='value')
        s.from_array(pair=('1','1'), r=r_arr, u=u_arr)
        u_actual = np.array([6.25,2.25,1])
        u = s.energy(pair=('1','1'), r=[1.5,2.5,3.5])
        np.testing.assert_allclose(u, u_actual)

        #test Spline with 2 knots
        s = relentless.potential.pair.Spline(types=('1',), num_knots=2, mode='value')
        s.from_array(pair=('1','1'), r=[1,2], u=[4,2])
        u = s.energy(pair=('1','1'), r=1.5)
        self.assertAlmostEqual(u, 3)

    def test_force(self):
        """Test force method"""
        r_arr = [1,2,3]
        u_arr = [9,4,1]

        #test diff mode
        s = relentless.potential.pair.Spline(types=('1',), num_knots=3)
        s.from_array(pair=('1','1'), r=r_arr, u=u_arr)
        f_actual = np.array([5,3,0])
        f = s.force(pair=('1','1'), r=[1.5,2.5,3.5])
        np.testing.assert_allclose(f, f_actual)

        #test value mode
        s = relentless.potential.pair.Spline(types=('1',), num_knots=3, mode='value')
        s.from_array(pair=('1','1'), r=r_arr, u=u_arr)
        f_actual = np.array([5,3,0])
        f = s.force(pair=('1','1'), r=[1.5,2.5,3.5])
        np.testing.assert_allclose(f, f_actual)

        #test Spline with 2 knots
        s = relentless.potential.pair.Spline(types=('1',), num_knots=2, mode='value')
        s.from_array(pair=('1','1'), r=[1,2], u=[4,2])
        f = s.force(pair=('1','1'), r=1.5)
        self.assertAlmostEqual(f, 2)

    def test_derivative(self):
        """Test derivative method"""
        r_arr = [1,2,3]
        u_arr = [9,4,1]

        #test diff mode
        s = relentless.potential.pair.Spline(types=('1',), num_knots=3)
        s.from_array(pair=('1','1'), r=r_arr, u=u_arr)
        d_actual = np.array([1.125,0.625,0])
        d = s.derivative(pair=('1','1'), param='knot-1', r=[1.5,2.5,3.5])
        np.testing.assert_allclose(d, d_actual)

        #test value mode
        s = relentless.potential.pair.Spline(types=('1',), num_knots=3, mode='value')
        s.from_array(pair=('1','1'), r=r_arr, u=u_arr)
        d_actual = np.array([0.75,0.75,0])
        d = s.derivative(pair=('1','1'), param='knot-1', r=[1.5,2.5,3.5])
        np.testing.assert_allclose(d, d_actual)

class test_Yukawa(unittest.TestCase):
    """Unit tests for relentless.potential.pair.Yukawa"""

    def test_init(self):
        """Test creation from data"""
        y = relentless.potential.pair.Yukawa(types=('1',))
        coeff = relentless.potential.potential.PairParameters(types=('1',), params=('epsilon','kappa','rmin','rmax','shift'))
        for pair in coeff.pairs:
            coeff[pair]['rmin'] = False
            coeff[pair]['rmax'] = False
            coeff[pair]['shift'] = False
        self.assertCountEqual(y.coeff.types, coeff.types)
        self.assertCountEqual(y.coeff.params, coeff.params)

    def test_energy(self):
        """Test _energy method"""
        y = relentless.potential.pair.Yukawa(types=('1',))

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
        y = relentless.potential.pair.Yukawa(types=('1',))

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
        y = relentless.potential.pair.Yukawa(types=('1',))

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

class test_Depletion(unittest.TestCase):
    """Unit tests for relentless.potential.pair.Depletion"""

    def test_init(self):
        """Test creation from data"""
        dp = relentless.potential.pair.Depletion(types=('1','2'))
        coeff = relentless.potential.potential.PairParameters(types=('1','2'),
                                                              params=('P','sigma_i','sigma_j','sigma_d','rmin','rmax','shift'))
        self.assertCountEqual(dp.coeff.types, coeff.types)
        self.assertCountEqual(dp.coeff.params, coeff.params)

    def test_energy(self):
        """Test _energy method"""
        dp = relentless.potential.pair.Depletion(types=('1',))

        #test scalar r
        r_input = 3
        u_actual = -4.6786414
        u = dp._energy(r=r_input, P=1, sigma_i=1.5, sigma_j=2, sigma_d=2.5)
        self.assertAlmostEqual(u, u_actual)

        #test array r
        r_input = np.array([1,1.75,4.25,5])
        u_actual = np.array([-25.25161952,-16.59621119,0,0])
        u = dp._energy(r=r_input, P=1, sigma_i=1.5, sigma_j=2, sigma_d=2.5)
        np.testing.assert_allclose(u, u_actual)

        #test negative sigma
        with self.assertRaises(ValueError):
            u = dp._energy(r=r_input, P=1, sigma_i=-1, sigma_j=1, sigma_d=1)
        with self.assertRaises(ValueError):
            u = dp._energy(r=r_input, P=1, sigma_i=1, sigma_j=-1, sigma_d=1)
        with self.assertRaises(ValueError):
            u = dp._energy(r=r_input, P=1, sigma_i=1, sigma_j=1, sigma_d=-1)

    def test_force(self):
        """Test _force method"""
        dp = relentless.potential.pair.Depletion(types=('1',))

        #test scalar r
        r_input = 3
        f_actual = -7.0682426
        f = dp._force(r=r_input, P=1, sigma_i=1.5, sigma_j=2, sigma_d=2.5)
        self.assertAlmostEqual(f, f_actual)

        #test array r
        r_input = np.array([1,1.75,4.25,5])
        f_actual = np.array([-11.54054444,-11.54054444,0,0])
        f = dp._force(r=r_input, P=1, sigma_i=1.5, sigma_j=2, sigma_d=2.5)
        np.testing.assert_allclose(f, f_actual)

        #test negative sigma
        with self.assertRaises(ValueError):
            f = dp._force(r=r_input, P=1, sigma_i=-1, sigma_j=1, sigma_d=1)
        with self.assertRaises(ValueError):
            f = dp._force(r=r_input, P=1, sigma_i=1, sigma_j=-1, sigma_d=1)
        with self.assertRaises(ValueError):
            f = dp._force(r=r_input, P=1, sigma_i=1, sigma_j=1, sigma_d=-1)

    def test_derivative(self):
        """Test _derivative method"""
        dp = relentless.potential.pair.Depletion(types=('1',))

        #w.r.t. P
        #test scalar r
        r_input = 3
        d_actual = -4.6786414
        d = dp._derivative(param='P', r=r_input, P=1, sigma_i=1.5, sigma_j=2, sigma_d=2.5)
        self.assertAlmostEqual(d, d_actual)

        #test array r
        r_input = np.array([1,1.75,4.25,5])
        d_actual = np.array([-25.25161952,-16.59621119,0,0])
        d = dp._derivative(param='P', r=r_input, P=1, sigma_i=1.5, sigma_j=2, sigma_d=2.5)
        np.testing.assert_allclose(d, d_actual)

        #w.r.t. sigma_i
        #test scalar r
        r_input = 3
        d_actual = -4.25424005
        d = dp._derivative(param='sigma_i', r=r_input, P=1, sigma_i=1.5, sigma_j=2, sigma_d=2.5)
        self.assertAlmostEqual(d, d_actual)

        #test array r
        r_input = np.array([1,1.75,4.25,5])
        d_actual = np.array([-11.242872,-8.975979,0,0])
        d = dp._derivative(param='sigma_i', r=r_input, P=1, sigma_i=1.5, sigma_j=2, sigma_d=2.5)
        np.testing.assert_allclose(d, d_actual)

        #w.r.t. sigma_j
        #test scalar r
        r_input = 3
        d_actual = -4.04970928
        d = dp._derivative(param='sigma_j', r=r_input, P=1, sigma_i=1.5, sigma_j=2, sigma_d=2.5)
        self.assertAlmostEqual(d, d_actual)

        #test array r
        r_input = np.array([1,1.75,4.25,5])
        d_actual = np.array([-8.397807,-7.573482,0,0])
        d = dp._derivative(param='sigma_j', r=r_input, P=1, sigma_i=1.5, sigma_j=2, sigma_d=2.5)
        np.testing.assert_allclose(d, d_actual)

        #w.r.t. sigma_d
        #test scalar r
        r_input = 3
        d_actual = -8.30394933
        d = dp._derivative(param='sigma_d', r=r_input, P=1, sigma_i=1.5, sigma_j=2, sigma_d=2.5)
        self.assertAlmostEqual(d, d_actual)

        #test array r
        r_input = np.array([1,1.75,4.25,5])
        d_actual = np.array([-21.454193,-16.549461,0,0])
        d = dp._derivative(param='sigma_d', r=r_input, P=1, sigma_i=1.5, sigma_j=2, sigma_d=2.5)
        np.testing.assert_allclose(d, d_actual)

        #test negative sigma
        with self.assertRaises(ValueError):
            d = dp._derivative(param='P', r=r_input, P=1, sigma_i=-1, sigma_j=1, sigma_d=1)
        with self.assertRaises(ValueError):
            d = dp._derivative(param='P', r=r_input, P=1, sigma_i=1, sigma_j=-1, sigma_d=1)
        with self.assertRaises(ValueError):
            d = dp._derivative(param='P', r=r_input, P=1, sigma_i=1, sigma_j=1, sigma_d=-1)

        #test invalid param
        with self.assertRaises(ValueError):
            d = dp._derivative(param='sigmaj', r=r_input, P=1, sigma_i=1, sigma_j=1, sigma_d=1)

if __name__ == '__main__':
    unittest.main()
