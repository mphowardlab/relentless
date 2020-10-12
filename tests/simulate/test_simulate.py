"""Unit tests for simulate module."""
import tempfile
import unittest

import numpy as np

import relentless

class QuadPot(relentless.potential.PairPotential):
    """Quadratic potential function used to test relentless.potential.Tabulator"""

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

class test_SimulationInstance(unittest.TestCase):
    """Unit tests for relentless.SimulationInstance"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.directory = relentless.Directory(self._tmp.name)

    def test_init(self):
        """Test creation from data."""
        options = {'constant_ens':True, 'constant_pot':False}
        ens = relentless.Ensemble(T=1.0, V=relentless.Cube(L=2.0), N={'A':2,'B':3})
        pots = relentless.simulate.Potentials()

        #no options
        sim = relentless.simulate.SimulationInstance(ens, pots, self.directory)
        self.assertEqual(sim.ensemble, ens)
        self.assertEqual(sim.potentials, pots)
        with self.assertRaises(AttributeError):
            sim.constant_ens

        #with options
        sim = relentless.simulate.SimulationInstance(ens, pots, self.directory, **options)
        self.assertEqual(sim.ensemble, ens)
        self.assertEqual(sim.potentials, pots)
        self.assertTrue(sim.constant_ens)
        self.assertFalse(sim.constant_pot)

    def tearDown(self):
        self._tmp.cleanup()

class test_PairPotentialTabulator(unittest.TestCase):
    """Unit tests for relentless.potential.Tabulator"""

    def test_init(self):
        """Test creation of object with data"""
        rs = np.array([0.0,0.5,1,1.5])

        #test creation with required param
        t = relentless.simulate.PairPotentialTabulator(rmax=1.5,num=4)
        np.testing.assert_allclose(t.r,rs)
        self.assertEqual(t.fmax, None)

        #test creation with required param, fmax, fcut, shift
        t = relentless.simulate.PairPotentialTabulator(rmax=1.5,num=4,fmax=1.5)
        np.testing.assert_allclose(t.r, rs)
        self.assertAlmostEqual(t.fmax, 1.5)

    def test_potential(self):
        """Test energy and force methods"""
        p1 = QuadPot(types=('1',), params=('m',))
        p1.coeff['1','1']['m'] = relentless.DesignVariable(2.0)
        p2 = QuadPot(types=('1','2'), params=('m',))
        for pair in p2.coeff.pairs:
            p2.coeff[pair]['m'] = 1.0
        t = relentless.simulate.PairPotentialTabulator(rmax=5,num=6,potentials=[p1,p2])

        # test energy method
        u = t.energy(('1','1'))
        np.testing.assert_allclose(u, np.array([27,12,3,0,3,12])-12)

        u = t.energy(('1','2'))
        np.testing.assert_allclose(u, np.array([9,4,1,0,1,4])-4)

        # test force method
        f = t.force(('1','1'))
        np.testing.assert_allclose(f, np.array([18,12,6,0,-6,-12]))

        f = t.force(('1','2'))
        np.testing.assert_allclose(f, np.array([6,4,2,0,-2,-4]))

        # test derivative method
        var = p1.coeff['1','1']['m']
        d = t.derivative(('1','1'),var)
        np.testing.assert_allclose(d, np.array([9,4,1,0,1,4])-4)

        d = t.derivative(('1','2'),var)
        np.testing.assert_allclose(d, np.array([0,0,0,0,0,0]))

    def test_fmax(self):
        """Test setting fmax"""
        p1 = QuadPot(types=('1',), params=('m',))
        p1.coeff['1','1']['m'] = 3.0

        t = relentless.simulate.PairPotentialTabulator(6.0, 7, p1, fmax=4)
        f = t.force(('1','1'))
        np.testing.assert_allclose(f, np.array([4,4,4,0,-4,-4,-4]))

        t.fmax = 12
        f = t.force(('1','1'))
        np.testing.assert_allclose(f, np.array([12,12,6,0,-6,-12,-12]))

        t.fmax = 20
        f = t.force(('1','1'))
        np.testing.assert_allclose(f, np.array([18,12,6,0,-6,-12,-18]))

if __name__ == '__main__':
    unittest.main()
