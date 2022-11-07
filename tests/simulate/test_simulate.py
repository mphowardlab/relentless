"""Unit tests for simulate module."""
from parameterized import parameterized_class
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
        if isinstance(m, relentless.model.DesignVariable):
            m = m.value
        u = m*(3-x)**2
        if s:
            u = u.item()
        return u

    def force(self, key, x):
        x,f,s = self._zeros(x)
        m = self.coeff[key]['m']
        if isinstance(m, relentless.model.DesignVariable):
            m = m.value
        f = 2*m*(3-x)
        if s:
            f = f.item()
        return f

    def derivative(self, key, var, x):
        x,d,s = self._zeros(x)
        if isinstance(var, relentless.model.DesignVariable):
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
        p1.coeff['1']['m'] = relentless.model.DesignVariable(2.0)
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
        p1.coeff['1','1']['m'] = relentless.model.DesignVariable(2.0)
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

@parameterized_class([
    {'box_geom': 'orthorhombic', 'dim': 3},
    {'box_geom': 'triclinic', 'dim': 3},
    {'box_geom': 'orthorhombic', 'dim': 2},
    {'box_geom': 'triclinic', 'dim': 2}],
    class_name_func=lambda cls, num, params_dict: "{}_{}_{}d".format(cls.__name__,params_dict['box_geom'],params_dict['dim']))
class test_InitializeRandomly(unittest.TestCase):
    def setUp(self):
        if self.box_geom == 'orthorhombic':
            if self.dim == 3:
                self.V = relentless.model.Cuboid(Lx=10, Ly=20, Lz=30)
            elif self.dim == 2:
                self.V = relentless.model.Rectangle(Lx=20, Ly=30)
        elif self.box_geom == 'triclinic':
            if self.dim == 3:
                self.V = relentless.model.TriclinicBox(Lx=10, Ly=20, Lz=30, xy=1, xz=2, yz=-1)
            elif self.dim == 2:
                self.V = relentless.model.ObliqueArea(Lx=20, Ly=30, xy=3)
        self.tol = 1.e-8

    def test_packing_one_type(self):
        if self.dim == 3:
            N = {'A': 150}
        else:
            N = {'A': 50}
        d = {'A': 1.2}
        rs, types = relentless.simulate.InitializeRandomly._pack_particles(42, N, self.V, d)

        self.assertTrue(all(typei == 'A' for typei in types))

        xs = self.V.coordinate_to_fraction(rs)
        self.assertTrue(numpy.all(xs >= 0))
        self.assertTrue(numpy.all(xs < 1))

        for i,r in enumerate(rs):
            dr = numpy.linalg.norm(rs[i+1:]-r, axis=1)
            self.assertTrue(numpy.all(dr > d['A']-self.tol))

    def test_packing_two_types(self):
        if self.dim == 3:
            N = {'A': 100, 'B': 25}
        else:
            N = {'A': 20, 'B': 5}
        d = {'A': 1.0, 'B': relentless.model.DesignVariable(3.0)}
        rs, types = relentless.simulate.InitializeRandomly._pack_particles(42, N, self.V, d)

        mask = numpy.array([typei == 'A' for typei in types])
        self.assertEqual(numpy.sum(mask), N['A'])
        self.assertEqual(numpy.sum(~mask), N['B'])

        xs = self.V.coordinate_to_fraction(rs)
        self.assertTrue(numpy.all(xs > -self.tol))
        self.assertTrue(numpy.all(xs < 1.+self.tol))

        rAs = rs[mask]
        rBs = rs[~mask]
        for i,rB in enumerate(rBs):
            dr = numpy.linalg.norm(rBs[i+1:]-rB, axis=1)
            self.assertTrue(numpy.all(dr > d['B'].value-self.tol))

        for i,rA in enumerate(rAs):
            dr = numpy.linalg.norm(rAs[i+1:]-rA, axis=1)
            self.assertTrue(numpy.all(dr > d['A']-self.tol))

        for i,rA in enumerate(rAs):
            dr = numpy.linalg.norm(rBs-rA,axis=1)
            dAB = 0.5*(d['A']+d['B'].value)
            if numpy.any(dr < dAB):
                print(dr[dr<dAB])
            self.assertTrue(numpy.all(dr > dAB-self.tol))

if __name__ == '__main__':
    unittest.main()
