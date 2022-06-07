"""Unit tests for potential module."""
import json
import tempfile
import unittest

import numpy

import relentless

class test_Parameters(unittest.TestCase):
    """Unit tests for relentless.potential.Parameters"""

    def test_init(self):
        """Test creation from data"""
        types = ('A','B')
        params = ('energy', 'mass')

        # test construction with tuple input
        m = relentless.potential.Parameters(types=('A','B'), params=('energy','mass'))
        self.assertEqual(m.types, types)
        self.assertEqual(m.params, params)

        # test construction with list input
        m = relentless.potential.Parameters(types=['A','B'], params=('energy','mass'))
        self.assertEqual(m.types, types)
        self.assertEqual(m.params, params)

        # test construction with mixed tuple/list input
        m = relentless.potential.Parameters(types=('A','B'), params=['energy','mass'])
        self.assertEqual(m.types, types)
        self.assertEqual(m.params, params)

        # test construction with invalid types
        with self.assertRaises(TypeError):
            m = relentless.potential.Parameters(types=(1,'B'), params=(1,2))

        # test construction with invalid parameters
        with self.assertRaises(TypeError):
            m = relentless.potential.Parameters(types=('A','B'), params=('1',2))

    def test_param_types(self):
        """Test various get and set methods on parameter types"""
        m = relentless.potential.Parameters(types=('A','B'), params=('energy', 'mass'))

        self.assertEqual(m.shared['energy'], None)
        self.assertEqual(m.shared['mass'], None)
        self.assertEqual(m['A']['energy'], None)
        self.assertEqual(m['A']['mass'], None)
        self.assertEqual(m['B']['energy'], None)
        self.assertEqual(m['B']['mass'], None)

        # test setting shared params
        m.shared.update(energy=1.0, mass=2.0)

        self.assertEqual(m.shared['energy'], 1.0)
        self.assertEqual(m.shared['mass'], 2.0)
        self.assertEqual(m['A']['energy'], None)
        self.assertEqual(m['A']['mass'], None)
        self.assertEqual(m['B']['energy'], None)
        self.assertEqual(m['B']['mass'], None)

        # test setting per-type params
        m['A'].update(energy=0.1, mass=0.2)
        m['B'].update(energy=0.2, mass=0.1)

        self.assertEqual(m.shared['energy'], 1.0)
        self.assertEqual(m.shared['mass'], 2.0)
        self.assertEqual(m['A']['energy'], 0.1)
        self.assertEqual(m['A']['mass'], 0.2)
        self.assertEqual(m['B']['energy'], 0.2)
        self.assertEqual(m['B']['mass'], 0.1)

    def test_accessor_methods(self):
        """Test various get and set methods on keys"""
        m = relentless.potential.Parameters(types=('A',), params=('energy','mass'))

        # test set and get for each pair type and param
        self.assertEqual(m['A']['energy'], None)
        self.assertEqual(m['A']['mass'], None)

        # test setting all key values at once
        m['A'] = {'energy':1.0, 'mass':2.0}
        self.assertEqual(m['A']['energy'], 1.0)
        self.assertEqual(m['A']['mass'], 2.0)

        # test setting key values partially with = operator
        m['A'] = {'energy':1.5}
        self.assertEqual(m['A']['energy'], 1.5)
        self.assertEqual(m['A']['mass'], None)

        m['A'] = {'mass':2.5}
        self.assertEqual(m['A']['energy'], None)
        self.assertEqual(m['A']['mass'], 2.5)

        # test setting key values partially using update()
        m['A'].update(energy=2.5) # using keyword arg
        self.assertEqual(m['A']['energy'], 2.5)
        self.assertEqual(m['A']['mass'], 2.5)

        m['A'].update({'mass':0.5}) # using dict
        self.assertEqual(m['A']['energy'], 2.5)
        self.assertEqual(m['A']['mass'], 0.5)

        # test accessing key param values individually
        m['A']['energy'] = 1.0
        self.assertEqual(m['A']['energy'], 1.0)
        self.assertEqual(m['A']['mass'], 0.5)

        m['A']['mass'] = 2.0
        self.assertEqual(m['A']['energy'], 1.0)
        self.assertEqual(m['A']['mass'], 2.0)

        # test reset and get via iteration
        params = ('energy', 'mass')
        for j in params:
            m['A'][j] = 1.5
        self.assertEqual(m['A']['energy'], 1.5)
        self.assertEqual(m['A']['mass'], 1.5)

        # test get and set on invalid keys/params
        with self.assertRaises(KeyError):
            m['C']['energy'] = 2
        with self.assertRaises(KeyError):
            x = m['C']['energy']
        with self.assertRaises(KeyError):
            m['A']['charge'] = 3
        with self.assertRaises(KeyError):
            x = m['A']['charge']

    def test_accessor_keys(self):
        """Test get and set methods for various keys"""
        # test set and get for initialized default
        m = relentless.potential.Parameters(types=('A','B'), params=('energy','mass'))
        self.assertEqual(m['A']['energy'], None)
        self.assertEqual(m['A']['mass'], None)
        self.assertEqual(m['B']['energy'], None)
        self.assertEqual(m['B']['mass'], None)

        # test reset and get manually
        m['A'] = {'energy':1.0, 'mass':2.0}
        self.assertEqual(m['A']['energy'], 1.0)
        self.assertEqual(m['A']['mass'], 2.0)
        self.assertEqual(m['B']['energy'], None)
        self.assertEqual(m['B']['mass'], None)

        m['A'].update({'energy':1.5, 'mass':2.5})
        self.assertEqual(m['A']['energy'], 1.5)
        self.assertEqual(m['A']['mass'], 2.5)
        self.assertEqual(m['B']['energy'], None)
        self.assertEqual(m['B']['mass'], None)

        m['B'] = {'energy':3.0, 'mass':4.0}
        self.assertEqual(m['A']['energy'], 1.5)
        self.assertEqual(m['A']['mass'], 2.5)
        self.assertEqual(m['B']['energy'], 3.0)
        self.assertEqual(m['B']['mass'], 4.0)

        m['B'].update(energy=3.5, mass=4.5)
        self.assertEqual(m['A']['energy'], 1.5)
        self.assertEqual(m['A']['mass'], 2.5)
        self.assertEqual(m['B']['energy'], 3.5)
        self.assertEqual(m['B']['mass'], 4.5)

        # test that partial assignment resets other param to default value
        m['A'] = {'energy':2.0}
        self.assertEqual(m['A']['energy'], 2.0)
        self.assertEqual(m['A']['mass'], None)
        self.assertEqual(m['B']['energy'], 3.5)
        self.assertEqual(m['B']['mass'], 4.5)

    def test_evaluate(self):
        """Test evaluation of keyed parameters"""
        # test evaluation with empty parameter values
        m = relentless.potential.Parameters(types=('A','B'), params=('energy','mass'))
        with self.assertRaises(ValueError):
            x = m.evaluate('A')

        # test evaluation with invalid key called
        with self.assertRaises(KeyError):
            x = m.evaluate('C')

        # test evaluation with initialized shared parameter values
        m = relentless.potential.Parameters(types=('A','B'), params=('energy','mass'))
        m.shared['energy'] = 0.0
        m.shared['mass'] = 0.5
        self.assertEqual(m.evaluate('A'), {'energy':0.0, 'mass':0.5})
        self.assertEqual(m.evaluate('B'), {'energy':0.0, 'mass':0.5})

        # test evaluation with initialized shared and individual parameter values
        m = relentless.potential.Parameters(types=('A','B'), params=('energy','mass'))
        m.shared['energy'] = 0.0
        m.shared['mass'] = 0.5
        m['B']['energy'] = 1.5
        self.assertEqual(m.evaluate('A'), {'energy':0.0, 'mass':0.5})
        self.assertEqual(m.evaluate('B'), {'energy':1.5, 'mass':0.5})

        # test evaluation with initialized parameter values as DesignVariable types
        m = relentless.potential.Parameters(types=('A',), params=('energy','mass'))
        m['A']['energy'] = relentless.variable.DesignVariable(value=-1.0,low=0.1)
        m['A']['mass'] = relentless.variable.DesignVariable(value=1.0,high=0.3)
        self.assertEqual(m.evaluate('A'), {'energy':0.1, 'mass':0.3})

        # test evaluation with initialized parameter values as unrecognized types
        m = relentless.potential.Parameters(types=('A','B'), params=('energy','mass'))
        m.shared['energy'] = relentless.math.Interpolator(x=(-1,0,1), y=(-2,0,2))
        m.shared['mass'] = relentless.math.Interpolator(x=(-1,0,1), y=(-2,0,2))
        with self.assertRaises(TypeError):
            x = m.evaluate('A')

    def test_save(self):
        """Test saving to file"""
        temp = tempfile.NamedTemporaryFile()

        # test dumping/re-loading data with scalar parameter values
        m = relentless.potential.Parameters(types=('A',), params=('energy','mass'))
        m['A']['energy'] = 1.5
        m['A']['mass'] = 2.5
        m.save(temp.name)
        with open(temp.name, 'r') as f:
            x = json.load(f)
        self.assertEqual(m['A']['energy'], x['A']['energy'])
        self.assertEqual(m['A']['mass'], x['A']['mass'])

        # test dumping/re-loading data with DesignVariable parameter values
        m = relentless.potential.Parameters(types=('A',), params=('energy','mass'))
        m['A']['energy'] = relentless.variable.DesignVariable(value=0.5)
        m['A']['mass'] = relentless.variable.DesignVariable(value=2.0)
        m.save(temp.name)
        with open(temp.name, 'r') as f:
            x = json.load(f)
        self.assertEqual(m['A']['energy'].value, x['A']['energy'])
        self.assertEqual(m['A']['mass'].value, x['A']['mass'])

        temp.close()

class MockPotential(relentless.potential.Potential):
    """Mock potential function used to test relentless.potential.Potential"""

    def __init__(self, types, params, container=None):
        super().__init__(types, params, container)

    def energy(self, key, x):
        pass

    def force(self, key, x):
        pass

    def derivative(self, key, var, x):
        pass

class MockContainer:
    """Mock container class used to test relentless.potential.Potential"""

    def __init__(self, types, params):
        self.types = ('foo')
        self.params = ('bar')

class test_Potential(unittest.TestCase):
    """Unit tests for relentless.potential.Potential"""

    def test_init(self):
        """Test creation from data"""
        # test creation with default container
        p = MockPotential(types=('1',), params=('m',))
        self.assertCountEqual(p.coeff.types, ('1',))
        self.assertCountEqual(p.coeff.params, ('m',))

        # test creation with custom container
        p = MockPotential(types=('1',), params=('m',), container=MockContainer)
        self.assertCountEqual(p.coeff.types, ('foo'))
        self.assertCountEqual(p.coeff.params, ('bar'))

    def test_zeros(self):
        """Test _zeros method"""
        p = MockPotential(types=('1',), params=('m',))

        # test with scalar r
        r = 0.5
        r_copy, u, s = p._zeros(r)
        numpy.testing.assert_allclose(r_copy, numpy.array([r]))
        numpy.testing.assert_allclose(u, numpy.zeros(1))
        self.assertEqual(s, True)

        u_copy = numpy.zeros(2)
        # test with list r
        r = [0.2, 0.3]
        r_copy, u, s = p._zeros(r)
        numpy.testing.assert_allclose(r_copy, r)
        numpy.testing.assert_allclose(u, u_copy)
        self.assertEqual(s, False)

        # test with tuple r
        r = (0.2, 0.3)
        r_copy, u, s = p._zeros(r)
        numpy.testing.assert_allclose(r_copy, r)
        numpy.testing.assert_allclose(u, u_copy)
        self.assertEqual(s, False)

        # test with numpy array r
        r = numpy.array([0.2, 0.3])
        r_copy, u, s = p._zeros(r)
        numpy.testing.assert_allclose(r_copy, r)
        numpy.testing.assert_allclose(u, u_copy)
        self.assertEqual(s, False)

        # test with 1-element array
        r = [0.2]
        r_copy, u, s = p._zeros(r)
        u_copy = numpy.zeros(1)
        numpy.testing.assert_allclose(r, r_copy)
        numpy.testing.assert_allclose(u, u_copy)
        self.assertEqual(s, False)

        # test with non 1-d array r
        r = numpy.array([[1, 2], [0.2, 0.3]])
        with self.assertRaises(TypeError):
            r_copy, u, s = p._zeros(r)

    def test_iteration(self):
        """Test iteration on Potential object"""
        p = MockPotential(types=('1','2'), params=('m','rmin','rmax'))
        for t in p.coeff:
            p.coeff[t]['m'] = 2.0
            p.coeff[t]['rmin'] = 0.0
            p.coeff[t]['rmax'] = 1.0

        self.assertEqual(dict(p.coeff['1']), {'m':2.0, 'rmin':0.0, 'rmax':1.0})
        self.assertEqual(dict(p.coeff['2']), {'m':2.0, 'rmin':0.0, 'rmax':1.0})

    def test_save(self):
        """Test saving to file"""
        temp = tempfile.NamedTemporaryFile()
        p = MockPotential(types=('1',), params=('m','rmin','rmax'))
        p.coeff['1']['m'] = 2.0
        p.coeff['1']['rmin'] = 0.0
        p.coeff['1']['rmax'] = 1.0

        p.save(temp.name)
        with open(temp.name, 'r') as f:
            x = json.load(f)

        self.assertEqual(p.coeff['1']['m'], x['1']['m'])
        self.assertEqual(p.coeff['1']['rmin'], x['1']['rmin'])
        self.assertEqual(p.coeff['1']['rmax'], x['1']['rmax'])

        temp.close()

if __name__ == '__main__':
    unittest.main()
