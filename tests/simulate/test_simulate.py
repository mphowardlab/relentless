"""Unit tests for simulate module."""
import tempfile
import unittest

import numpy

import relentless

class QuadPot(relentless.potential.Potential):
    """Quadratic potential function used to test relentless.simulate.PotentialTabulator"""

    def __init__(self, types, params):
        super().__init__(types, params)

    def energy(self, key, x):
        x,u,s = self._zeros(x)
        m = self.coeff[key]['m']
        if isinstance(m, relentless.variable.DesignVariable):
            m = m.value
        u = m*(3-x)**2
        if s:
            u = u.item()
        return u

    def force(self, key, x):
        x,f,s = self._zeros(x)
        m = self.coeff[key]['m']
        if isinstance(m, relentless.variable.DesignVariable):
            m = m.value
        f = 2*m*(3-x)
        if s:
            f = f.item()
        return f

    def derivative(self, key, var, x):
        x,d,s = self._zeros(x)
        if isinstance(var, relentless.variable.DesignVariable):
            if self.coeff[key]['m'] is var:
                d = (3-x)**2
        if s:
            d = d.item()
        return d

class test_PotentialTabulator(unittest.TestCase):
    """Unit tests for relentless.simulate.PotentialTabulator"""

    def test_init(self):
        """Test creation with data."""
        xs = numpy.array([0.0,0.5,1,1.5])
        p1 = QuadPot(types=('1',), params=('m',))

        # test creation with no potential
        t = relentless.simulate.PotentialTabulator(start=0.0,stop=1.5,num=4)
        numpy.testing.assert_allclose(t.x,xs)
        self.assertAlmostEqual(t.start, 0.0)
        self.assertAlmostEqual(t.stop, 1.5)
        self.assertEqual(t.num, 4)
        self.assertEqual(t.potentials, [])

        # test creation with defined potential
        t = relentless.simulate.PotentialTabulator(start=0.0,stop=1.5,num=4,potentials=p1)
        numpy.testing.assert_allclose(t.x,xs)
        self.assertAlmostEqual(t.start, 0.0)
        self.assertAlmostEqual(t.stop, 1.5)
        self.assertEqual(t.num, 4)
        self.assertEqual(t.potentials, [p1])

    def test_potential(self):
        """Test energy, force, and derivative methods"""
        p1 = QuadPot(types=('1',), params=('m',))
        p1.coeff['1']['m'] = relentless.variable.DesignVariable(2.0)
        p2 = QuadPot(types=('1','2'), params=('m',))
        for key in p2.coeff.types:
            p2.coeff[key]['m'] = 1.0
        t = relentless.simulate.PotentialTabulator(start=0.0,stop=5.0,num=6,potentials=[p1,p2])

        # test energy method
        u = t.energy('1')
        numpy.testing.assert_allclose(u, numpy.array([27,12,3,0,3,12]))

        u = t.energy('2')
        numpy.testing.assert_allclose(u, numpy.array([9,4,1,0,1,4]))

        # test force method
        f = t.force('1')
        numpy.testing.assert_allclose(f, numpy.array([18,12,6,0,-6,-12]))

        f = t.force('2')
        numpy.testing.assert_allclose(f, numpy.array([6,4,2,0,-2,-4]))

        # test derivative method
        var = p1.coeff['1']['m']
        d = t.derivative('1',var)
        numpy.testing.assert_allclose(d, numpy.array([9,4,1,0,1,4]))

        d = t.derivative('2',var)
        numpy.testing.assert_allclose(d, numpy.array([0,0,0,0,0,0]))

class QuadPairPot(relentless.potential.PairPotential):
    """Quadratic pair potential function used to test relentless.simulate.PairPotentialTabulator"""

    def __init__(self, types, params):
        super().__init__(types, params)

    def _energy(self, r, m, **params):
        r,u,s = self._zeros(r)
        u = m*(3-r)**2
        if s:
            u = u.item()
        return u

    def _force(self, r, m, **params):
        r,f,s = self._zeros(r)
        f = 2*m*(3-r)
        if s:
            f = f.item()
        return f

    def _derivative(self, param, r, **params):
        r,d,s = self._zeros(r)
        if param == 'm':
            d = (3-r)**2
        if s:
            d = d.item()
        return d

class test_PairPotentialTabulator(unittest.TestCase):
    """Unit tests for relentless.simulate.PairPotentialTabulator"""

    def test_init(self):
        """Test creation with data."""
        rs = numpy.array([0.0,0.5,1,1.5])

        # test creation with only required parameters
        t = relentless.simulate.PairPotentialTabulator(rmin=0.0,rmax=1.5,num=4,neighbor_buffer=0.4)
        numpy.testing.assert_allclose(t.r,rs)
        self.assertAlmostEqual(t.rmin, 0.0)
        self.assertAlmostEqual(t.rmax, 1.5)
        self.assertEqual(t.num, 4)
        self.assertEqual(t.neighbor_buffer, 0.4)
        self.assertEqual(t.fmax, None)

        # test creation with required parameters and fmax
        t = relentless.simulate.PairPotentialTabulator(rmin=0.0,rmax=1.5,num=4,neighbor_buffer=0.4,fmax=1.5)
        numpy.testing.assert_allclose(t.r, rs)
        self.assertAlmostEqual(t.rmin, 0.0)
        self.assertAlmostEqual(t.rmax, 1.5)
        self.assertEqual(t.num, 4)
        self.assertEqual(t.neighbor_buffer, 0.4)
        self.assertAlmostEqual(t.fmax, 1.5)

    def test_potential(self):
        """Test energy, force, and derivative methods"""
        p1 = QuadPairPot(types=('1',), params=('m',))
        p1.coeff['1','1']['m'] = relentless.variable.DesignVariable(2.0)
        p2 = QuadPairPot(types=('1','2'), params=('m',))
        for pair in p2.coeff.pairs:
            p2.coeff[pair]['m'] = 1.0
        t = relentless.simulate.PairPotentialTabulator(rmin=0,rmax=5,num=6,neighbor_buffer=0.4,potentials=[p1,p2])

        # test energy method
        u = t.energy(('1','1'))
        numpy.testing.assert_allclose(u, numpy.array([27,12,3,0,3,12])-12)

        u = t.energy(('1','2'))
        numpy.testing.assert_allclose(u, numpy.array([9,4,1,0,1,4])-4)

        # test force method
        f = t.force(('1','1'))
        numpy.testing.assert_allclose(f, numpy.array([18,12,6,0,-6,-12]))

        f = t.force(('1','2'))
        numpy.testing.assert_allclose(f, numpy.array([6,4,2,0,-2,-4]))

        # test derivative method
        var = p1.coeff['1','1']['m']
        d = t.derivative(('1','1'),var)
        numpy.testing.assert_allclose(d, numpy.array([9,4,1,0,1,4])-4)

        d = t.derivative(('1','2'),var)
        numpy.testing.assert_allclose(d, numpy.array([0,0,0,0,0,0]))

    def test_fmax(self):
        """Test setting and changing fmax."""
        p1 = QuadPairPot(types=('1',), params=('m',))
        p1.coeff['1','1']['m'] = 3.0

        t = relentless.simulate.PairPotentialTabulator(rmin=0.0,rmax=6.0,num=7,neighbor_buffer=0.4,potentials=p1,fmax=4)
        f = t.force(('1','1'))
        numpy.testing.assert_allclose(f, numpy.array([4,4,4,0,-4,-4,-4]))

        t.fmax = 12
        f = t.force(('1','1'))
        numpy.testing.assert_allclose(f, numpy.array([12,12,6,0,-6,-12,-12]))

        t.fmax = 20
        f = t.force(('1','1'))
        numpy.testing.assert_allclose(f, numpy.array([18,12,6,0,-6,-12,-18]))

if __name__ == '__main__':
    unittest.main()
