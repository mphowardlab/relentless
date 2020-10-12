"""Unit tests for potential module."""
import json
import tempfile
import unittest

import numpy as np

import relentless

class test_PairParameters(unittest.TestCase):
    """Unit tests for relentless.potential.PairParameters"""

    def test_init(self):
        """Test creation from data"""
        types = ('A','B')
        pairs = (('A','B'), ('B','B'), ('A','A'))
        params = ('energy', 'mass')

        #test construction with tuple input
        m = relentless.potential.PairParameters(types=('A','B'), params=('energy','mass'))
        self.assertEqual(m.types, types)
        self.assertEqual(m.params, params)
        self.assertCountEqual(m.pairs, pairs)

        #test construction with list input
        m = relentless.potential.PairParameters(types=['A','B'], params=('energy','mass'))
        self.assertEqual(m.types, types)
        self.assertEqual(m.params, params)
        self.assertCountEqual(m.pairs, pairs)

        #test construction with mixed tuple/list input
        m = relentless.potential.PairParameters(types=('A','B'), params=['energy','mass'])
        self.assertEqual(m.types, types)
        self.assertEqual(m.params, params)
        self.assertCountEqual(m.pairs, pairs)

        #test construction with int type parameters
        with self.assertRaises(TypeError):
            m = relentless.potential.PairParameters(types=('A','B'), params=(1,2))

        #test construction with mixed type parameters
        with self.assertRaises(TypeError):
            m = relentless.potential.PairParameters(types=('A','B'), params=('1',2))

    def test_param_types(self):
        """Test various get and set methods on pair parameter types"""
        m = relentless.potential.PairParameters(types=('A','B'), params=('energy', 'mass'))

        self.assertEqual(m.shared['energy'], None)
        self.assertEqual(m.shared['mass'], None)
        self.assertEqual(m['A','A']['energy'], None)
        self.assertEqual(m['A','A']['mass'], None)
        self.assertEqual(m['A','B']['energy'], None)
        self.assertEqual(m['A','B']['mass'], None)
        self.assertEqual(m['B','B']['energy'], None)
        self.assertEqual(m['B','B']['mass'], None)
        self.assertEqual(m['A']['energy'], None)
        self.assertEqual(m['A']['mass'], None)
        self.assertEqual(m['B']['energy'], None)
        self.assertEqual(m['B']['mass'], None)

        #test setting shared params
        m.shared.update(energy=1.0, mass=2.0)

        self.assertEqual(m.shared['energy'], 1.0)
        self.assertEqual(m.shared['mass'], 2.0)
        self.assertEqual(m['A','A']['energy'], None)
        self.assertEqual(m['A','A']['mass'], None)
        self.assertEqual(m['A','B']['energy'], None)
        self.assertEqual(m['A','B']['mass'], None)
        self.assertEqual(m['B','B']['energy'], None)
        self.assertEqual(m['B','B']['mass'], None)
        self.assertEqual(m['A']['energy'], None)
        self.assertEqual(m['A']['mass'], None)
        self.assertEqual(m['B']['energy'], None)
        self.assertEqual(m['B']['mass'], None)

        #test setting per-pair params
        m['A','A'].update(energy=1.5, mass=2.5)
        m['A','B'].update(energy=2.0, mass=3.0)
        m['B','B'].update(energy=0.5, mass=0.7)

        self.assertEqual(m.shared['energy'], 1.0)
        self.assertEqual(m.shared['mass'], 2.0)
        self.assertEqual(m['A','A']['energy'], 1.5)
        self.assertEqual(m['A','A']['mass'], 2.5)
        self.assertEqual(m['A','B']['energy'], 2.0)
        self.assertEqual(m['A','B']['mass'], 3.0)
        self.assertEqual(m['B','B']['energy'], 0.5)
        self.assertEqual(m['B','B']['mass'], 0.7)
        self.assertEqual(m['A']['energy'], None)
        self.assertEqual(m['A']['mass'], None)
        self.assertEqual(m['B']['energy'], None)
        self.assertEqual(m['B']['mass'], None)

        #test setting per-type params
        m['A'].update(energy=0.1, mass=0.2)
        m['B'].update(energy=0.2, mass=0.1)

        self.assertEqual(m.shared['energy'], 1.0)
        self.assertEqual(m.shared['mass'], 2.0)
        self.assertEqual(m['A','A']['energy'], 1.5)
        self.assertEqual(m['A','A']['mass'], 2.5)
        self.assertEqual(m['A','B']['energy'], 2.0)
        self.assertEqual(m['A','B']['mass'], 3.0)
        self.assertEqual(m['B','B']['energy'], 0.5)
        self.assertEqual(m['B','B']['mass'], 0.7)
        self.assertEqual(m['A']['energy'], 0.1)
        self.assertEqual(m['A']['mass'], 0.2)
        self.assertEqual(m['B']['energy'], 0.2)
        self.assertEqual(m['B']['mass'], 0.1)

    def test_accessor_methods(self):
        """Test various get and set methods on pairs"""
        m = relentless.potential.PairParameters(types=('A',), params=('energy','mass'))

        #test set and get for each pair type and param
        self.assertEqual(m['A','A']['energy'], None)
        self.assertEqual(m['A','A']['mass'], None)

        #test setting all key values at once
        m['A','A'] = {'energy':1.0, 'mass':2.0}
        self.assertEqual(m['A','A']['energy'], 1.0)
        self.assertEqual(m['A','A']['mass'], 2.0)

        #test setting key values partially with = operator
        m['A','A'] = {'energy':1.5}
        self.assertEqual(m['A','A']['energy'], 1.5)
        self.assertEqual(m['A','A']['mass'], None)

        m['A','A'] = {'mass':2.5}
        self.assertEqual(m['A','A']['energy'], None)
        self.assertEqual(m['A','A']['mass'], 2.5)

        #test setting key values partially using update()
        m['A','A'].update(energy=2.5)  #using keyword arg
        self.assertEqual(m['A','A']['energy'], 2.5)
        self.assertEqual(m['A','A']['mass'], 2.5)

        m['A','A'].update({'mass':0.5})  #using dict
        self.assertEqual(m['A','A']['energy'], 2.5)
        self.assertEqual(m['A','A']['mass'], 0.5)

        #test accessing key param values individually
        m['A','A']['energy'] = 1.0
        self.assertEqual(m['A','A']['energy'], 1.0)
        self.assertEqual(m['A','A']['mass'], 0.5)

        m['A','A']['mass'] = 2.0
        self.assertEqual(m['A','A']['energy'], 1.0)
        self.assertEqual(m['A','A']['mass'], 2.0)

        #test reset and get via iteration
        params = ('energy', 'mass')
        for j in params:
            m['A','A'][j] = 1.5
        self.assertEqual(m['A','A']['energy'], 1.5)
        self.assertEqual(m['A','A']['mass'], 1.5)

        #test get and set on invalid pairs/params
        with self.assertRaises(KeyError):
            m['A','C']['energy'] = 2
        with self.assertRaises(KeyError):
            x = m['A','C']['energy']
        with self.assertRaises(KeyError):
            m['A','B']['charge'] = 3
        with self.assertRaises(KeyError):
            x = m['A','B']['charge']

    def test_accessor_pairs(self):
        """Test get and set methods for various pairs"""
        #test set and get for initialized default
        m = relentless.potential.PairParameters(types=('A','B'), params=('energy','mass'))
        self.assertEqual(m['A','B']['energy'], None)
        self.assertEqual(m['A','B']['mass'], None)
        self.assertEqual(m['A','A']['energy'], None)
        self.assertEqual(m['A','A']['mass'], None)
        self.assertEqual(m['B','B']['energy'], None)
        self.assertEqual(m['B','B']['mass'], None)

        #test reset and get manually
        m['A','B'] = {'energy':1.0, 'mass':2.0}
        self.assertEqual(m['A','B']['energy'], 1.0)
        self.assertEqual(m['A','B']['mass'], 2.0)
        self.assertEqual(m['A','A']['energy'], None)
        self.assertEqual(m['A','A']['mass'], None)
        self.assertEqual(m['B','B']['energy'], None)
        self.assertEqual(m['B','B']['mass'], None)

        m['A','A'].update({'energy':1.5, 'mass':2.5})
        self.assertEqual(m['A','B']['energy'], 1.0)
        self.assertEqual(m['A','B']['mass'], 2.0)
        self.assertEqual(m['A','A']['energy'], 1.5)
        self.assertEqual(m['A','A']['mass'], 2.5)
        self.assertEqual(m['B','B']['energy'], None)
        self.assertEqual(m['B','B']['mass'], None)

        m['B','B'] = {'energy':3.0, 'mass':4.0}
        self.assertEqual(m['A','B']['energy'], 1.0)
        self.assertEqual(m['A','B']['mass'], 2.0)
        self.assertEqual(m['A','A']['energy'], 1.5)
        self.assertEqual(m['A','A']['mass'], 2.5)
        self.assertEqual(m['B','B']['energy'], 3.0)
        self.assertEqual(m['B','B']['mass'], 4.0)

        m['B','B'].update(energy=3.5, mass=4.5)
        self.assertEqual(m['A','B']['energy'], 1.0)
        self.assertEqual(m['A','B']['mass'], 2.0)
        self.assertEqual(m['A','A']['energy'], 1.5)
        self.assertEqual(m['A','A']['mass'], 2.5)
        self.assertEqual(m['B','B']['energy'], 3.5)
        self.assertEqual(m['B','B']['mass'], 4.5)

        #test that partial assignment resets other param to default value
        m['A','A'] = {'energy':2.0}
        self.assertEqual(m['A','B']['energy'], 1.0)
        self.assertEqual(m['A','B']['mass'], 2.0)
        self.assertEqual(m['A','A']['energy'], 2.0)
        self.assertEqual(m['A','A']['mass'], None)
        self.assertEqual(m['B','B']['energy'], 3.5)
        self.assertEqual(m['B','B']['mass'], 4.5)

    def test_evaluate(self):
        """Test evaluation of pair parameters"""
        #test evaluation with empty parameter values
        m = relentless.potential.PairParameters(types=('A','B'), params=('energy','mass'))
        with self.assertRaises(ValueError):
            x = m.evaluate(('A','B'))

        #test evaluation with invalid pair called
        with self.assertRaises(KeyError):
            x = m.evaluate(('A','C'))

        #test evaluation with initialized shared parameter values
        m = relentless.potential.PairParameters(types=('A','B'), params=('energy','mass'))
        m.shared['energy'] = 0.0
        m.shared['mass'] = 0.5
        self.assertEqual(m.evaluate(('A','B')), {'energy':0.0, 'mass':0.5})
        self.assertEqual(m.evaluate(('A','A')), {'energy':0.0, 'mass':0.5})
        self.assertEqual(m.evaluate(('B','B')), {'energy':0.0, 'mass':0.5})
        self.assertEqual(m.evaluate(('B','A')), m.evaluate(('A','B')))

        #test evaluation with initialized shared and individual parameter values
        m = relentless.potential.PairParameters(types=('A','B'), params=('energy','mass'))
        m.shared['energy'] = 0.0
        m.shared['mass'] = 0.5
        m['B','B']['energy'] = 1.5
        self.assertEqual(m.evaluate(('A','B')), {'energy':0.0, 'mass':0.5})
        self.assertEqual(m.evaluate(('A','A')), {'energy':0.0, 'mass':0.5})
        self.assertEqual(m.evaluate(('B','B')), {'energy':1.5, 'mass':0.5})
        self.assertEqual(m.evaluate(('B','A')), m.evaluate(('A','B')))

        #test evaluation with initialized parameter values as DesignVariable types
        m = relentless.potential.PairParameters(types=('A',), params=('energy','mass'))
        m['A','A']['energy'] = relentless.DesignVariable(value=-1.0,low=0.1)
        m['A','A']['mass'] = relentless.DesignVariable(value=1.0,high=0.3)
        self.assertEqual(m.evaluate(('A','A')), {'energy':0.1, 'mass':0.3})

        #test evaluation with initialized parameter values as unrecognized types
        m = relentless.potential.PairParameters(types=('A','B'), params=('energy','mass'))
        m.shared['energy'] = relentless.Interpolator(x=(-1,0,1), y=(-2,0,2))
        m.shared['mass'] = relentless.Interpolator(x=(-1,0,1), y=(-2,0,2))
        with self.assertRaises(TypeError):
            x = m.evaluate(('A','B'))

    def test_save(self):
        """Test saving to file"""
        temp = tempfile.NamedTemporaryFile()

        #test dumping/re-loading data with scalar parameter values
        m = relentless.potential.PairParameters(types=('A',), params=('energy','mass'))
        m['A','A']['energy'] = 1.5
        m['A','A']['mass'] = 2.5
        m.save(temp.name)
        with open(temp.name, 'r') as f:
            x = json.load(f)
        self.assertEqual(m['A','A']['energy'], x["('A', 'A')"]['energy'])
        self.assertEqual(m['A','A']['mass'], x["('A', 'A')"]['mass'])

        #test dumping/re-loading data with DesignVariable parameter values
        m = relentless.potential.PairParameters(types=('A',), params=('energy','mass'))
        m['A','A']['energy'] = relentless.DesignVariable(value=0.5)
        m['A','A']['mass'] = relentless.DesignVariable(value=2.0)
        m.save(temp.name)
        with open(temp.name, 'r') as f:
            x = json.load(f)
        self.assertEqual(m['A','A']['energy'].value, x["('A', 'A')"]['energy'])
        self.assertEqual(m['A','A']['mass'].value, x["('A', 'A')"]['mass'])

        temp.close()

    def test_design_variables(self):
        """Test design_variables method."""
        m = relentless.potential.PairParameters(types=('A',), params=('energy','mass','charge'))

        #test complex dependent variables as parameters
        a = relentless.DesignVariable(value=1.0)
        b = relentless.DesignVariable(value=2.0)
        c = relentless.SameAs(a=b)
        d = relentless.ArithmeticMean(a=b, b=c)

        m['A','A']['energy'] = 1.5
        m['A','A']['mass'] = a
        m['A','A']['charge'] = d

        x = m.design_variables()
        self.assertCountEqual(x, (a,b))

        #test same variable as multiple parameters
        m['A','A']['energy'] = a
        m['A','A']['mass'] = a
        m['A','A']['charge'] = a

        x = m.design_variables()
        self.assertCountEqual(x, (a,))

class LinPot(relentless.potential.PairPotential):
    """Linear potential function used to test relentless.potential.PairPotential"""

    def __init__(self, types, params):
        super().__init__(types, params)

    def _energy(self, r, m, **params):
        r,u,s = self._zeros(r)
        u[:] = m*r
        if s:
            u = u.item()
        return u

    def _force(self, r, m, **params):
        r,f,s = self._zeros(r)
        f[:] = -m
        if s:
            f = f.item()
        return f

    def _derivative(self, param, r, **params):
        r,d,s = self._zeros(r)
        if param == 'm':
            d[:] = r
        if s:
            d = d.item()
        return d

class TwoVarPot(relentless.potential.PairPotential):
    """Mock potential function used to test relentless.potential.PairPotential.derivative"""

    def __init__(self, types, params):
        super().__init__(types, params)

    def _energy(self, r, x, y, **params):
        pass

    def _force(self, r, x, y, **params):
        pass

    def _derivative(self, param, r, **params):
        #not real derivative, just used to test functionality
        r,d,s = self._zeros(r)
        if param == 'x':
            d[:] = 2*r
        elif param == 'y':
            d[:] = 3*r
        if s:
            d = d.item()
        return d

class test_PairPotential(unittest.TestCase):
    """Unit tests for relentless.potential.PairPotential"""

    def test_init(self):
        """Test creation from data"""
        #test creation with only m
        p = LinPot(types=('1',), params=('m',))
        p.coeff['1','1']['m'] = 3.5

        coeff = relentless.potential.PairParameters(types=('1',), params=('m','rmin','rmax','shift'))
        coeff['1','1']['m'] = 3.5
        coeff['1','1']['rmin'] = False
        coeff['1','1']['rmax'] = False
        coeff['1','1']['shift'] = False

        self.assertCountEqual(p.coeff.types, coeff.types)
        self.assertCountEqual(p.coeff.params, coeff.params)
        self.assertDictEqual(p.coeff.evaluate(('1','1')), coeff.evaluate(('1','1')))

        #test creation with m and rmin
        p = LinPot(types=('1',), params=('m','rmin'))
        p.coeff['1','1']['m'] = 3.5
        p.coeff['1','1']['rmin'] = 0.0

        coeff = relentless.potential.PairParameters(types=('1',), params=('m','rmin','rmax','shift'))
        coeff['1','1']['m'] = 3.5
        coeff['1','1']['rmin'] = 0.0
        coeff['1','1']['rmax'] = False
        coeff['1','1']['shift'] = False

        self.assertCountEqual(p.coeff.types, coeff.types)
        self.assertCountEqual(p.coeff.params, coeff.params)
        self.assertDictEqual(p.coeff.evaluate(('1','1')), coeff.evaluate(('1','1')))

        #test creation with m and rmax
        p = LinPot(types=('1',), params=('m','rmax'))
        p.coeff['1','1']['m'] = 3.5
        p.coeff['1','1']['rmax'] = 1.0

        coeff = relentless.potential.PairParameters(types=('1',), params=('m','rmin','rmax','shift'))
        coeff['1','1']['m'] = 3.5
        coeff['1','1']['rmin'] = False
        coeff['1','1']['rmax'] = 1.0
        coeff['1','1']['shift'] = False

        self.assertCountEqual(p.coeff.types, coeff.types)
        self.assertCountEqual(p.coeff.params, coeff.params)
        self.assertDictEqual(p.coeff.evaluate(('1','1')), coeff.evaluate(('1','1')))

        #test creation with m and shift
        p = LinPot(types=('1',), params=('m','shift'))
        p.coeff['1','1']['m'] = 3.5
        p.coeff['1','1']['shift'] = True

        coeff = relentless.potential.PairParameters(types=('1',), params=('m','rmin','rmax','shift'))
        coeff['1','1']['m'] = 3.5
        coeff['1','1']['rmin'] = False
        coeff['1','1']['rmax'] = False
        coeff['1','1']['shift'] = True

        self.assertCountEqual(p.coeff.types, coeff.types)
        self.assertCountEqual(p.coeff.params, coeff.params)
        self.assertDictEqual(p.coeff.evaluate(('1','1')), coeff.evaluate(('1','1')))

        #test creation with all params
        p = LinPot(types=('1',), params=('m','rmin','rmax','shift'))
        p.coeff['1','1']['m'] = 3.5
        p.coeff['1','1']['rmin'] = 0.0
        p.coeff['1','1']['rmax'] = 1.0
        p.coeff['1','1']['shift'] = True

        coeff = relentless.potential.PairParameters(types=('1',), params=('m','rmin','rmax','shift'))
        coeff['1','1']['m'] = 3.5
        coeff['1','1']['rmin'] = 0.0
        coeff['1','1']['rmax'] = 1.0
        coeff['1','1']['shift'] = True

        self.assertCountEqual(p.coeff.types, coeff.types)
        self.assertCountEqual(p.coeff.params, coeff.params)
        self.assertDictEqual(p.coeff.evaluate(('1','1')), coeff.evaluate(('1','1')))

    def test_zeros(self):
        """Test _zeros method"""
        p = LinPot(types=('1',), params=('m',))

        #test with scalar r
        r = 0.5
        r_copy, u, s = p._zeros(r)
        np.testing.assert_allclose(r_copy, np.array([r]))
        np.testing.assert_allclose(u, np.zeros(1))
        self.assertEqual(s, True)

        u_copy = np.zeros(2)
        #test with list r
        r = [0.2, 0.3]
        r_copy, u, s = p._zeros(r)
        np.testing.assert_allclose(r_copy, r)
        np.testing.assert_allclose(u, u_copy)
        self.assertEqual(s, False)

        #test with tuple r
        r = (0.2, 0.3)
        r_copy, u, s = p._zeros(r)
        np.testing.assert_allclose(r_copy, r)
        np.testing.assert_allclose(u, u_copy)
        self.assertEqual(s, False)

        #test with numpy array r
        r = np.array([0.2, 0.3])
        r_copy, u, s = p._zeros(r)
        np.testing.assert_allclose(r_copy, r)
        np.testing.assert_allclose(u, u_copy)
        self.assertEqual(s, False)

        #test with 1-element array
        r = [0.2]
        r_copy, u, s = p._zeros(r)
        u_copy = np.zeros(1)
        np.testing.assert_allclose(r, r_copy)
        np.testing.assert_allclose(u, u_copy)
        self.assertEqual(s, False)

        #test with non 1-d array r
        r = np.array([[1, 2], [0.2, 0.3]])
        with self.assertRaises(TypeError):
            r_copy, u, s = p._zeros(r)

    def test_energy(self):
        """Test energy method"""
        p = LinPot(types=('1',), params=('m',))
        p.coeff['1','1']['m'] = 2.0

        #test with no cutoffs
        u = p.energy(pair=('1','1'), r=0.5)
        self.assertAlmostEqual(u, 1.0)
        u = p.energy(pair=('1','1'), r=[0.25,0.75])
        np.testing.assert_allclose(u, [0.5,1.5])

        #test with rmin set
        p.coeff['1','1']['rmin'] = 0.5
        u = p.energy(pair=('1','1'), r=0.6)
        self.assertAlmostEqual(u, 1.2)
        u = p.energy(pair=('1','1'), r=[0.25,0.75])
        np.testing.assert_allclose(u, [1.0,1.5])

        #test with rmax set
        p.coeff['1','1'].update(rmin=False, rmax=1.5)
        u = p.energy(pair=('1','1'), r=1.0)
        self.assertAlmostEqual(u, 2.0)
        u = p.energy(pair=('1','1'), r=[0.25,1.75])
        np.testing.assert_allclose(u, [0.5,3.0])

        #test with rmin and rmax set
        p.coeff['1','1']['rmin'] = 0.5
        u = p.energy(pair=('1','1'), r=0.75)
        self.assertAlmostEqual(u, 1.5)
        u = p.energy(pair=('1','1'), r=[0.25,0.5,1.5,1.75])
        np.testing.assert_allclose(u, [1.0,1.0,3.0,3.0])

        #test with shift set
        p.coeff['1','1'].update(shift=True)
        u = p.energy(pair=('1','1'), r=0.5)
        self.assertAlmostEqual(u, -2.0)
        u = p.energy(pair=('1','1'), r=[0.25,0.75,1.0,1.5])
        np.testing.assert_allclose(u, [-2.0,-1.5,-1.0,0.0])

        #test with shift set without rmax
        p.coeff['1','1'].update(rmax=False)
        with self.assertRaises(ValueError):
            u = p.energy(pair=('1','1'), r=0.5)

    def test_force(self):
        """Test force method"""
        p = LinPot(types=('1',), params=('m',))
        p.coeff['1','1']['m'] = 2.0

        #test with no cutoffs
        f = p.force(pair=('1','1'), r=0.5)
        self.assertAlmostEqual(f, -2.0)
        f = p.force(pair=('1','1'), r=[0.25,0.75])
        np.testing.assert_allclose(f, [-2.0,-2.0])

        #test with rmin set
        p.coeff['1','1']['rmin'] = 0.5
        f = p.force(pair=('1','1'), r=0.6)
        self.assertAlmostEqual(f, -2.0)
        f = p.force(pair=('1','1'), r=[0.25,0.75])
        np.testing.assert_allclose(f, [0.0,-2.0])

        #test with rmax set
        p.coeff['1','1'].update(rmin=False, rmax=1.5)
        f = p.force(pair=('1','1'), r=1.0)
        self.assertAlmostEqual(f, -2.0)
        f = p.force(pair=('1','1'), r=[0.25,1.75])
        np.testing.assert_allclose(f, [-2.0,0.0])

        #test with rmin and rmax set
        p.coeff['1','1']['rmin'] = 0.5
        f = p.force(pair=('1','1'), r=0.75)
        self.assertAlmostEqual(f, -2.0)
        f = p.force(pair=('1','1'), r=[0.25,0.5,1.5,1.75])
        np.testing.assert_allclose(f, [0.0,-2.0,-2.0,0.0])

        #test with shift set
        p.coeff['1','1'].update(shift=True)
        f = p.force(pair=('1','1'), r=0.5)
        self.assertAlmostEqual(f, -2.0)
        f = p.force(pair=('1','1'), r=[1.0,1.5])
        np.testing.assert_allclose(f, [-2.0,-2.0])

    def test_derivative_values(self):
        """Test derivative method with different param values"""
        p = LinPot(types=('1',), params=('m',))
        x = relentless.DesignVariable(value=2.0)
        p.coeff['1','1']['m'] = x

        #test with no cutoffs
        d = p.derivative(pair=('1','1'), var=x, r=0.5)
        self.assertAlmostEqual(d, 0.5)
        d = p.derivative(pair=('1','1'), var=x, r=[0.25,0.75])
        np.testing.assert_allclose(d, [0.25,0.75])

        #test with rmin set
        rmin = relentless.DesignVariable(value=0.5)
        p.coeff['1','1']['rmin'] = rmin
        d = p.derivative(pair=('1','1'), var=x, r=0.6)
        self.assertAlmostEqual(d, 0.6)
        d = p.derivative(pair=('1','1'), var=x, r=[0.25,0.75])
        np.testing.assert_allclose(d, [0.5,0.75])

        #test with rmax set
        rmax = relentless.DesignVariable(value=1.5)
        p.coeff['1','1'].update(rmin=False, rmax=rmax)
        d = p.derivative(pair=('1','1'), var=x, r=1.0)
        self.assertAlmostEqual(d, 1.0)
        d = p.derivative(pair=('1','1'), var=x, r=[0.25,1.75])
        np.testing.assert_allclose(d, [0.25,1.5])

        #test with rmin and rmax set
        p.coeff['1','1']['rmin'] = rmin
        d = p.derivative(pair=('1','1'), var=x, r=0.75)
        self.assertAlmostEqual(d, 0.75)
        d = p.derivative(pair=('1','1'), var=x, r=[0.25,0.5,1.5,1.75])
        np.testing.assert_allclose(d, [0.5,0.5,1.5,1.5])

        #test w.r.t. rmin and rmax
        d = p.derivative(pair=('1','1'), var=rmin, r=[0.25,1.0,2.0])
        np.testing.assert_allclose(d, [2.0,0.0,0.0])
        d = p.derivative(pair=('1','1'), var=rmax, r=[0.25,1.0,2.0])
        np.testing.assert_allclose(d, [0.0,0.0,2.0])

        #test parameter derivative with shift set
        p.coeff['1','1'].update(shift=True)
        d = p.derivative(pair=('1','1'), var=x, r=0.5)
        self.assertAlmostEqual(d, -1.0)
        d = p.derivative(pair=('1','1'), var=x, r=[0.25,1.0,1.5,1.75])
        np.testing.assert_allclose(d, [-1.0,-0.5,0.0,0.0])

        #test w.r.t. rmin and rmax, shift set
        d = p.derivative(pair=('1','1'), var=rmin, r=[0.25,1.0,2.0])
        np.testing.assert_allclose(d, [2.0,0.0,0.0])
        d = p.derivative(pair=('1','1'), var=rmax, r=[0.25,1.0,2.0])
        np.testing.assert_allclose(d, [-2.0,-2.0,0.0])

    def test_derivative_types(self):
        """Test derivative method with different param types."""
        q = LinPot(types=('1',), params=('m',))
        x = relentless.DesignVariable(value=4.0)
        y = relentless.DesignVariable(value=64.0)
        z = relentless.GeometricMean(x, y)
        q.coeff['1','1']['m'] = z

        #test with respect to dependent variable parameter
        d = q.derivative(pair=('1','1'), var=z, r=2.0)
        self.assertAlmostEqual(d, 2.0)

        #test with respect to independent variable on which parameter is dependent
        d = q.derivative(pair=('1','1'), var=x, r=1.5)
        self.assertAlmostEqual(d, 3.0)
        d = q.derivative(pair=('1','1'), var=y, r=4.0)
        self.assertAlmostEqual(d, 0.5)

        #test invalid derivative w.r.t. scalar
        a = 2.5
        q.coeff['1','1']['m'] = a
        with self.assertRaises(TypeError):
            d = q.derivative(pair=('1','1'), var=a, r=2.0)

        #test with respect to independent variable which is related to a SameAs variable
        r = TwoVarPot(types=('1',), params=('x','y'))

        r.coeff['1','1']['x'] = x
        r.coeff['1','1']['y'] = relentless.SameAs(x)
        d = r.derivative(pair=('1','1'), var=x, r=4.0)
        self.assertAlmostEqual(d, 20.0)

        r.coeff['1','1']['y'] = x
        r.coeff['1','1']['x'] = relentless.SameAs(x)
        d = r.derivative(pair=('1','1'), var=x, r=4.0)
        self.assertAlmostEqual(d, 20.0)

    def test_iteration(self):
        """Test iteration on PairPotential object"""
        p = LinPot(types=('1','2'), params=('m',))
        for pair in p.coeff:
            p.coeff[pair]['m'] = 2.0
            p.coeff[pair]['rmin'] = 0.0
            p.coeff[pair]['rmax'] = 1.0

        self.assertDictEqual(p.coeff['1','1'].todict(), {'m':2.0, 'rmin':0.0, 'rmax':1.0, 'shift':False})
        self.assertDictEqual(p.coeff['1','2'].todict(), {'m':2.0, 'rmin':0.0, 'rmax':1.0, 'shift':False})
        self.assertDictEqual(p.coeff['2','2'].todict(), {'m':2.0, 'rmin':0.0, 'rmax':1.0, 'shift':False})

    def test_save(self):
        """Test saving to file"""
        temp = tempfile.NamedTemporaryFile()
        p = LinPot(types=('1',), params=('m','rmin','rmax'))
        p.coeff['1','1']['m'] = 2.0
        p.coeff['1','1']['rmin'] = 0.0
        p.coeff['1','1']['rmax'] = 1.0
        p.coeff['1','1']['shift'] = True

        p.save(temp.name)
        with open(temp.name, 'r') as f:
            x = json.load(f)

        self.assertEqual(p.coeff['1','1']['m'], x["('1', '1')"]['m'])
        self.assertEqual(p.coeff['1','1']['rmin'], x["('1', '1')"]['rmin'])
        self.assertEqual(p.coeff['1','1']['rmax'], x["('1', '1')"]['rmax'])
        self.assertEqual(p.coeff['1','1']['shift'], x["('1', '1')"]['shift'])

        temp.close()

if __name__ == '__main__':
    unittest.main()
