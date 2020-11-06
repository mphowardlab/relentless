"""Unit tests for simulate module."""
import tempfile
import unittest

import numpy as np

import relentless

class test_Simulation(unittest.TestCase):
    """Unit tests for relentless.Simulation"""

    #mock functions for use as operations
    class CheckEnsemble(relentless.simulate.SimulationOperation):
        def __call__(self, sim):
            try:
                sim[self].value = sim.constant_ens
            except AttributeError:
                sim[self].value = False

    class CheckPotential(relentless.simulate.SimulationOperation):
        def __call__(self, sim):
            try:
                sim[self].value = sim.constant_pot
            except AttributeError:
                sim[self].value = False

    def test_init(self):
        """Test creation from data."""
        operations = [self.CheckEnsemble(), self.CheckPotential()]
        options = {'constant_ens':True, 'constant_pot':True}

        #no operations, no options
        d = relentless.simulate.Simulation()
        self.assertCountEqual(d.operations, [])
        self.assertDictEqual(d.options, {})

        #with operations, no options
        d = relentless.simulate.Simulation(operations)
        self.assertCountEqual(d.operations, operations)
        self.assertDictEqual(d.options, {})

        #no operations, with options
        d = relentless.simulate.Simulation(**options)
        self.assertCountEqual(d.operations, [])
        self.assertDictEqual(d.options, options)

        #with operations, with options
        d = relentless.simulate.Simulation(operations, **options)
        self.assertCountEqual(d.operations, operations)
        self.assertDictEqual(d.options, options)

    def test_run(self):
        """Test run method."""
        ens = relentless.Ensemble(T=1.0, P=4.0, N={'A':2})
        pot = relentless.PairMatrix(types=ens.types)
        for pair in pot:
            pot[pair]['r'] = np.array([1.,2.,3.])
            pot[pair]['u'] = np.array([2.,3.,4.])
        dirc = 'mock'
        sim = relentless.simulate.Simulation()
        operations = [self.CheckEnsemble(), self.CheckPotential()]
        options = {'constant_ens':True, 'constant_pot':True}

        #no operations, no options
        sim = relentless.simulate.Simulation()
        sim_ = sim.run(ensemble=ens, potentials=pot, directory=dirc)
        with self.assertRaises(AttributeError):
            sim_[operations[0]].value
        with self.assertRaises(AttributeError):
            sim_[operations[1]].value

        #with operations, no options
        sim = relentless.simulate.Simulation(operations=operations)
        sim_ = sim.run(ensemble=ens, potentials=pot, directory=dirc)
        self.assertFalse(sim_[operations[0]].value)
        self.assertFalse(sim_[operations[1]].value)

        #with operations, options
        sim = relentless.simulate.Simulation(operations=operations, **options)
        sim_ = sim.run(ensemble=ens, potentials=pot, directory=dirc)
        self.assertTrue(sim_[operations[0]].value)
        self.assertTrue(sim_[operations[1]].value)

        #invalid operation type
        sim = relentless.simulate.Simulation(operations='a')
        with self.assertRaises(TypeError):
            sim.run(ensemble=ens, potential=pot, directory=dirc)

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
        sim = relentless.simulate.SimulationInstance(None, ens, pots, self.directory)
        self.assertEqual(sim.ensemble, ens)
        self.assertEqual(sim.potentials, pots)
        with self.assertRaises(AttributeError):
            sim.constant_ens

        #with options
        sim = relentless.simulate.SimulationInstance(None, ens, pots, self.directory, **options)
        self.assertEqual(sim.ensemble, ens)
        self.assertEqual(sim.potentials, pots)
        self.assertTrue(sim.constant_ens)
        self.assertFalse(sim.constant_pot)

    def tearDown(self):
        self._tmp.cleanup()

class QuadPot(relentless.potential.Potential):
    """Quadratic potential function used to test relentless.simulate.PotentialTabulator"""

    def __init__(self, types, params):
        super().__init__(types, params)

    def energy(self, key, x):
        x,u,s = self._zeros(x)
        m = self.coeff[key]['m']
        if isinstance(m, relentless.DesignVariable):
            m = m.value
        u = m*(3-x)**2
        if s:
            u = u.item()
        return u

    def force(self, key, x):
        x,f,s = self._zeros(x)
        m = self.coeff[key]['m']
        if isinstance(m, relentless.DesignVariable):
            m = m.value
        f = 2*m*(3-x)
        if s:
            f = f.item()
        return f

    def derivative(self, key, var, x):
        x,d,s = self._zeros(x)
        if isinstance(var, relentless.DesignVariable):
            if self.coeff[key]['m'] is var:
                d = (3-x)**2
        if s:
            d = d.item()
        return d

class test_PotentialTabulator(unittest.TestCase):
    """Unit tests for relentless.simulate.PotentialTabulator"""

    def test_init(self):
        """Test creation with data."""
        xs = np.array([0.0,0.5,1,1.5])
        p1 = QuadPot(types=('1',), params=('m',))

        #test creation with no potential
        t = relentless.simulate.PotentialTabulator(start=0.0,stop=1.5,num=4)
        np.testing.assert_allclose(t.x,xs)
        self.assertAlmostEqual(t.start, 0.0)
        self.assertAlmostEqual(t.stop, 1.5)
        self.assertEqual(t.num, 4)
        self.assertCountEqual(t.potentials, [])

        #test creation with defined potential
        t = relentless.simulate.PotentialTabulator(start=0.0,stop=1.5,num=4,potentials=p1)
        np.testing.assert_allclose(t.x,xs)
        self.assertAlmostEqual(t.start, 0.0)
        self.assertAlmostEqual(t.stop, 1.5)
        self.assertEqual(t.num, 4)
        self.assertCountEqual(t.potentials, [p1])

    def test_potential(self):
        """Test energy, force, and derivative methods"""
        p1 = QuadPot(types=('1',), params=('m',))
        p1.coeff['1']['m'] = relentless.DesignVariable(2.0)
        p2 = QuadPot(types=('1','2'), params=('m',))
        for key in p2.coeff.types:
            p2.coeff[key]['m'] = 1.0
        t = relentless.simulate.PotentialTabulator(start=0.0,stop=5.0,num=6,potentials=[p1,p2])

        # test energy method
        u = t.energy('1')
        np.testing.assert_allclose(u, np.array([27,12,3,0,3,12]))

        u = t.energy('2')
        np.testing.assert_allclose(u, np.array([9,4,1,0,1,4]))

        # test force method
        f = t.force('1')
        np.testing.assert_allclose(f, np.array([18,12,6,0,-6,-12]))

        f = t.force('2')
        np.testing.assert_allclose(f, np.array([6,4,2,0,-2,-4]))

        # test derivative method
        var = p1.coeff['1']['m']
        d = t.derivative('1',var)
        np.testing.assert_allclose(d, np.array([9,4,1,0,1,4]))

        d = t.derivative('2',var)
        np.testing.assert_allclose(d, np.array([0,0,0,0,0,0]))

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
        rs = np.array([0.0,0.5,1,1.5])

        #test creation with required param
        t = relentless.simulate.PairPotentialTabulator(rmax=1.5,num=4)
        np.testing.assert_allclose(t.r,rs)
        self.assertAlmostEqual(t.rmax, 1.5)
        self.assertEqual(t.num, 4)
        self.assertEqual(t.fmax, None)

        #test creation with required param, fmax, fcut, shift
        t = relentless.simulate.PairPotentialTabulator(rmax=1.5,num=4,fmax=1.5)
        np.testing.assert_allclose(t.r, rs)
        self.assertAlmostEqual(t.rmax, 1.5)
        self.assertEqual(t.num, 4)
        self.assertAlmostEqual(t.fmax, 1.5)

    def test_potential(self):
        """Test energy, force, and derivative methods"""
        p1 = QuadPairPot(types=('1',), params=('m',))
        p1.coeff['1','1']['m'] = relentless.DesignVariable(2.0)
        p2 = QuadPairPot(types=('1','2'), params=('m',))
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
        """Test setting and changing fmax."""
        p1 = QuadPairPot(types=('1',), params=('m',))
        p1.coeff['1','1']['m'] = 3.0

        t = relentless.simulate.PairPotentialTabulator(rmax=6.0,num=7,potentials=p1,fmax=4)
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
