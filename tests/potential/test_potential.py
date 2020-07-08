"""Unit tests for potential module."""
import unittest
import tempfile
import numpy as np

from relentless import core
from relentless.potential import potential

class test_CoefficientMatrix(unittest.TestCase):
    """Unit tests for potential.CoefficientMatrix"""

    def test_init(self):
        """Test creation from data"""
        types = ('A','B')
        pairs = (('A','B'), ('B','B'), ('A','A'))
        params = ('energy', 'mass')

        #test construction with tuple input
        m = potential.CoefficientMatrix(types=('A','B'), params=('energy','mass'))
        self.assertEqual(m.types, types)
        self.assertEqual(m.params, params)
        self.assertCountEqual(m.pairs, pairs)

        #test construction with list input
        m = potential.CoefficientMatrix(types=['A','B'], params=('energy','mass'))
        self.assertEqual(m.types, types)
        self.assertEqual(m.params, params)
        self.assertCountEqual(m.pairs, pairs)

        #test construction with mixed tuple/list input
        m = potential.CoefficientMatrix(types=('A','B'), params=['energy','mass'])
        self.assertEqual(m.types, types)
        self.assertEqual(m.params, params)
        self.assertCountEqual(m.pairs, pairs)

        #test construction with default values initialized
        default = {'energy':0.0, 'mass':0.0}
        m = potential.CoefficientMatrix(types=('A','B'), params=('energy','mass'),
                                        default={'energy':0.0, 'mass':0.0})
        self.assertEqual(m.types, types)
        self.assertEqual(m.params, params)
        self.assertEqual(m.default.todict(), default)
        self.assertCountEqual(m.pairs, pairs)

        #test construction with int type parameters
        with self.assertRaises(TypeError):
            m = potential.CoefficientMatrix(types=('A','B'), params=(1,2))

        #test construction with mixed type parameters
        with self.assertRaises(TypeError):
            m = potential.CoefficientMatrix(types=('A','B'), params=('1',2))

    def test_accessor_methods(self):
        """Test various get and set methods on pairs"""
        m = potential.CoefficientMatrix(types=('A',), params=('energy','mass'))

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
        m = potential.CoefficientMatrix(types=('A','B'), params=('energy','mass'),
                                        default={'energy':0.0, 'mass':0.0})
        self.assertEqual(m['A','B']['energy'], 0.0)
        self.assertEqual(m['A','B']['mass'], 0.0)
        self.assertEqual(m['A','A']['energy'], 0.0)
        self.assertEqual(m['A','A']['mass'], 0.0)
        self.assertEqual(m['B','B']['energy'], 0.0)
        self.assertEqual(m['B','B']['mass'], 0.0)

        #test reset and get manually
        m['A','B'] = {'energy':1.0, 'mass':2.0}
        self.assertEqual(m['A','B']['energy'], 1.0)
        self.assertEqual(m['A','B']['mass'], 2.0)
        self.assertEqual(m['A','A']['energy'], 0.0)
        self.assertEqual(m['A','A']['mass'], 0.0)
        self.assertEqual(m['B','B']['energy'], 0.0)
        self.assertEqual(m['B','B']['mass'], 0.0)

        m['A','A'].update({'energy':1.5, 'mass':2.5})
        self.assertEqual(m['A','B']['energy'], 1.0)
        self.assertEqual(m['A','B']['mass'], 2.0)
        self.assertEqual(m['A','A']['energy'], 1.5)
        self.assertEqual(m['A','A']['mass'], 2.5)
        self.assertEqual(m['B','B']['energy'], 0.0)
        self.assertEqual(m['B','B']['mass'], 0.0)

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
        self.assertEqual(m['A','A']['mass'], 0.0)
        self.assertEqual(m['B','B']['energy'], 3.5)
        self.assertEqual(m['B','B']['mass'], 4.5)

    def test_evaluate(self):
        """Test evaluation of pair parameters"""
        #test evaluation with empty parameter values
        m = potential.CoefficientMatrix(types=('A','B'), params=('energy','mass'))
        with self.assertRaises(ValueError):
            x = m.evaluate(('A','B'))

        #test evaluation with invalid pair called
        with self.assertRaises(KeyError):
            x = m.evaluate(('A','C'))

        #test evaluation with initialized parameter values as scalars
        m = potential.CoefficientMatrix(types=('A','B'), params=('energy','mass'),
                                        default={'energy':0.0, 'mass':0.0})
        self.assertEqual(m.evaluate(('A','B')), {'energy':0.0, 'mass':0.0})
        self.assertEqual(m.evaluate(('A','A')), {'energy':0.0, 'mass':0.0})
        self.assertEqual(m.evaluate(('B','B')), {'energy':0.0, 'mass':0.0})
        self.assertEqual(m.evaluate(('B','A')), m.evaluate(('A','B')))

        #test evaluation with initialized parameter values as Variable types
        m = potential.CoefficientMatrix(types=('A','B'), params=('energy','mass'),
                                        default={'energy':core.Variable(value=-1.0,low=0.1),
                                                 'mass':core.Variable(value=1.0,high=0.1)})
        self.assertEqual(m.evaluate(('A','B')), {'energy':0.1, 'mass':0.1})
        self.assertEqual(m.evaluate(('A','A')), {'energy':0.1, 'mass':0.1})
        self.assertEqual(m.evaluate(('B','B')), {'energy':0.1, 'mass':0.1})
        self.assertEqual(m.evaluate(('B','A')), m.evaluate(('A','B')))

        #test evaluation with initialized parameter values as unrecognized types
        m = potential.CoefficientMatrix(types=('A','B'), params=('energy','mass'),
                                        default={'energy':core.Interpolator(x=(-1,0,1), y=(-2,0,2)),
                                                 'mass':core.Interpolator(x=(-1,0,1), y=(-2,0,2))})
        with self.assertRaises(TypeError):
            x = m.evaluate(('A','B'))

    def test_saveload(self):
        """Test saving to and loading from file"""
        temp = tempfile.NamedTemporaryFile()

        #test dumping/re-loading data with scalar parameter values
        m = potential.CoefficientMatrix(types=('A','B'), params=('energy','mass'),
                                        default={'energy':0.5, 'mass':2.0})
        x = potential.CoefficientMatrix(types=('A','B'), params=('energy','mass'))
        m.save(temp.name)
        x.load(temp.name)

        self.assertEqual(m['A','B']['energy'], x['A','B']['energy'])
        self.assertEqual(m['A','B']['mass'], x['A','B']['mass'])
        self.assertEqual(m['A','A']['energy'], x['A','A']['energy'])
        self.assertEqual(m['A','A']['mass'], x['A','A']['mass'])
        self.assertEqual(m['B','B']['energy'], x['B','B']['energy'])
        self.assertEqual(m['B','B']['mass'], x['B','B']['mass'])

        #test dumping/re-loading data with Variable parameter values
        m = potential.CoefficientMatrix(types=('A','B'), params=('energy','mass'),
                                        default={'energy':core.Variable(value=0.5, high=0.2),
                                                 'mass':core.Variable(value=2.0, low=3.0)})
        x = potential.CoefficientMatrix(types=('A','B'), params=('energy','mass'))
        m.save(temp.name)
        x.load(temp.name)

        self.assertEqual(m['A','B']['energy'].value, x['A','B']['energy'].value)
        self.assertEqual(m['A','B']['mass'].value, x['A','B']['mass'].value)
        self.assertEqual(m['A','A']['energy'].value, x['A','A']['energy'].value)
        self.assertEqual(m['A','A']['mass'].value, x['A','A']['mass'].value)
        self.assertEqual(m['B','B']['energy'].value, x['B','B']['energy'].value)
        self.assertEqual(m['B','B']['mass'].value, x['B','B']['mass'].value)

        self.assertEqual(m['A','B']['energy'].low, x['A','B']['energy'].low)
        self.assertEqual(m['A','B']['mass'].low, x['A','B']['mass'].low)
        self.assertEqual(m['A','A']['energy'].low, x['A','A']['energy'].low)
        self.assertEqual(m['A','A']['mass'].low, x['A','A']['mass'].low)
        self.assertEqual(m['B','B']['energy'].low, x['B','B']['energy'].low)
        self.assertEqual(m['B','B']['mass'].low, x['B','B']['mass'].low)

        self.assertEqual(m['A','B']['energy'].high, x['A','B']['energy'].high)
        self.assertEqual(m['A','B']['mass'].high, x['A','B']['mass'].high)
        self.assertEqual(m['A','A']['energy'].high, x['A','A']['energy'].high)
        self.assertEqual(m['A','A']['mass'].high, x['A','A']['mass'].high)
        self.assertEqual(m['B','B']['energy'].high, x['B','B']['energy'].high)
        self.assertEqual(m['B','B']['mass'].high, x['B','B']['mass'].high)

        #test dumping/re-loading data with types that don't match
        m = potential.CoefficientMatrix(types=('A','B'), params=('energy','mass'))
        x = potential.CoefficientMatrix(types=('A','B','C'), params=('energy','mass'),
                                          default={'energy':0.5, 'mass':0.2})
        x.save(temp.name)
        with self.assertRaises(KeyError):
            m.load(temp.name)

        #test dumping/re-loading data with params that don't match
        x = potential.CoefficientMatrix(types=('A','B'), params=('energy','mass','charge'),
                                          default={'energy':core.Variable(value=1.0),
                                                   'mass':core.Variable(value=2.0),
                                                   'charge':core.Variable(value=0.0)})
        x.save(temp.name)
        with self.assertRaises(KeyError):
            m.load(temp.name)

        temp.close()

    def test_fromfile(self):
        """Test creating new coefficient matrix from file."""
        temp = tempfile.NamedTemporaryFile()

        #test loading data with scalar values using class method `from_file`
        x = potential.CoefficientMatrix(types=('A','B'), params=('energy','mass'),
                                          default={'energy':1.0, 'mass':2.0})
        x.save(temp.name)
        m = potential.CoefficientMatrix.from_file(temp.name)

        self.assertEqual(m['A','B']['energy'], x['A','B']['energy'])
        self.assertEqual(m['A','B']['mass'], x['A','B']['mass'])
        self.assertEqual(m['A','A']['energy'], x['A','A']['energy'])
        self.assertEqual(m['A','A']['mass'], x['A','A']['mass'])
        self.assertEqual(m['B','B']['energy'], x['B','B']['energy'])
        self.assertEqual(m['B','B']['mass'], x['B','B']['mass'])

        #test loading data with Variable values using class method `from_file`
        x = potential.CoefficientMatrix(types=('A','B'), params=('energy','mass'),
                                          default={'energy':core.Variable(value=1.0,low=1.5),
                                                   'mass':core.Variable(value=0.5,high=0.2)})
        x.save(temp.name)
        m = potential.CoefficientMatrix.from_file(temp.name)

        self.assertEqual(m['A','B']['energy'].value, x['A','B']['energy'].value)
        self.assertEqual(m['A','B']['mass'].value, x['A','B']['mass'].value)
        self.assertEqual(m['A','A']['energy'].value, x['A','A']['energy'].value)
        self.assertEqual(m['A','A']['mass'].value, x['A','A']['mass'].value)
        self.assertEqual(m['B','B']['energy'].value, x['B','B']['energy'].value)
        self.assertEqual(m['B','B']['mass'].value, x['B','B']['mass'].value)

        self.assertEqual(m['A','B']['energy'].low, x['A','B']['energy'].low)
        self.assertEqual(m['A','B']['mass'].low, x['A','B']['mass'].low)
        self.assertEqual(m['A','A']['energy'].low, x['A','A']['energy'].low)
        self.assertEqual(m['A','A']['mass'].low, x['A','A']['mass'].low)
        self.assertEqual(m['B','B']['energy'].low, x['B','B']['energy'].low)
        self.assertEqual(m['B','B']['mass'].low, x['B','B']['mass'].low)

        self.assertEqual(m['A','B']['energy'].high, x['A','B']['energy'].high)
        self.assertEqual(m['A','B']['mass'].high, x['A','B']['mass'].high)
        self.assertEqual(m['A','A']['energy'].high, x['A','A']['energy'].high)
        self.assertEqual(m['A','A']['mass'].high, x['A','A']['mass'].high)
        self.assertEqual(m['B','B']['energy'].high, x['B','B']['energy'].high)
        self.assertEqual(m['B','B']['mass'].high, x['B','B']['mass'].high)

        temp.close()

class test_PairPotential(unittest.TestCase):
    """Unit tests for potential.PairPotential"""

    class LinPot(potential.PairPotential):
        """Linear potential function used to test potential.PairPotential"""
        def __init__(self, types, params, default={}):
            super().__init__(types, params, default)

        def _energy(self, r, m, **params):
            r,u,s = self._zeros(r)
            u = m*r
            if s:
                u = u.item()
            return u

        def _force(self, r, m, **params):
            r,f,s = self._zeros(r)
            f = -m
            if s:
                f = f.item()
            return f

        def _derivative(self, param, r, **params):
            r,d,s = self._zeros(r)
            if param == 'm':
                d = r
            if s:
                d = d.item()
            return d

    def test_init(self):
        """Test creation from data"""
        #test creation with only m
        p = test_PairPotential.LinPot(types=('1',), params=('m',),
                                      default={'m':3.5})
        coeff = potential.CoefficientMatrix(types=('1',), params=('m','rmin','rmax','shift'),
                                            default={'m':3.5,'rmin':False,'rmax':False,'shift':False})
        self.assertCountEqual(p.coeff.types, coeff.types)
        self.assertCountEqual(p.coeff.params, coeff.params)
        self.assertDictEqual(p.coeff.default.todict(), coeff.default.todict())

        #test creation with m and rmin
        p = test_PairPotential.LinPot(types=('1',), params=('m','rmin'),
                                      default={'m':3.5,'rmin':0.0})
        coeff = potential.CoefficientMatrix(types=('1',), params=('m','rmin','rmax','shift'),
                                            default={'m':3.5,'rmin':0.0,'rmax':False,'shift':False})
        self.assertCountEqual(p.coeff.types, coeff.types)
        self.assertCountEqual(p.coeff.params, coeff.params)
        self.assertDictEqual(p.coeff.default.todict(), coeff.default.todict())

        #test creation with m and rmax
        p = test_PairPotential.LinPot(types=('1',), params=('m','rmax'),
                                      default={'m':3.5,'rmax':1.0})
        coeff = potential.CoefficientMatrix(types=('1',), params=('m','rmin','rmax','shift'),
                                            default={'m':3.5,'rmin':False,'rmax':1.0,'shift':False})
        self.assertCountEqual(p.coeff.types, coeff.types)
        self.assertCountEqual(p.coeff.params, coeff.params)
        self.assertDictEqual(p.coeff.default.todict(), coeff.default.todict())

        #test creation with m and shift
        p = test_PairPotential.LinPot(types=('1',), params=('m','shift'),
                                      default={'m':3.5,'shift':True})
        coeff = potential.CoefficientMatrix(types=('1',), params=('m','rmin','rmax','shift'),
                                            default={'m':3.5,'rmin':False,'rmax':False,'shift':True})
        self.assertCountEqual(p.coeff.types, coeff.types)
        self.assertCountEqual(p.coeff.params, coeff.params)
        self.assertDictEqual(p.coeff.default.todict(), coeff.default.todict())

        #test creation with all params, default values
        p = test_PairPotential.LinPot(types=('1',), params=('m','rmin','rmax','shift'),
                                      default={'m':3.5,'rmin':0.0,'rmax':1.0,'shift':True})
        coeff = potential.CoefficientMatrix(types=('1',), params=('m','rmin','rmax','shift'),
                                            default={'m':3.5,'rmin':0.0,'rmax':1.0,'shift':True})
        self.assertCountEqual(p.coeff.types, coeff.types)
        self.assertCountEqual(p.coeff.params, coeff.params)
        self.assertDictEqual(p.coeff.default.todict(), coeff.default.todict())

        #test creation with all params, no default values
        p = test_PairPotential.LinPot(types=('1',), params=('m',))
        coeff = potential.CoefficientMatrix(types=('1',), params=('m','rmin','rmax','shift'),
                                            default={'rmin':False,'rmax':False,'shift':False})
        self.assertCountEqual(p.coeff.types, coeff.types)
        self.assertCountEqual(p.coeff.params, coeff.params)
        self.assertDictEqual(p.coeff.default.todict(), coeff.default.todict())

    def test_zeros(self):
        """Test _zeros method"""
        p = test_PairPotential.LinPot(types=('1',), params=('m',))

        #test with scalar r
        r = 0.5
        r_copy, u, s = p._zeros(r)
        self.assertEqual(r_copy, r)
        self.assertEqual(u, 0)
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

        #test with non 1-d array r
        r = np.array([[1, 2], [0.2, 0.3]])
        with self.assertRaises(TypeError):
            r_copy, u, s = p._zeros(r)

    def test_energy(self):
        """Test energy method"""
        #test with no cutoffs
        p = test_PairPotential.LinPot(types=('1',), params=('m',),
                                      default={'m':2})
        u = p.energy(pair=(('1','1')), r=0.5)
        self.assertEqual(u, 1.0)
        u = p.energy(pair=(('1','1')), r=[0.25,0.75])
        np.testing.assert_allclose(u, [0.5,1.5])

        #test with rmin set
        p = test_PairPotential.LinPot(types=('1',), params=('m','rmin'),
                                      default={'m':2,'rmin':0.0})
        u = p.energy(pair=(('1','1')), r=0.0)
        self.assertEqual(u, 0.0)
        u = p.energy(pair=(('1','1')), r=[-0.25,0.75])
        np.testing.assert_allclose(u, [0.0,1.5])

        #test with rmax set
        p = test_PairPotential.LinPot(types=('1',), params=('m','rmax'),
                                      default={'m':2,'rmax':1.0})
        u = p.energy(pair=(('1','1')), r=1.0)
        self.assertEqual(u, 2.0)
        u = p.energy(pair=(('1','1')), r=[0.25,1.75])
        np.testing.assert_allclose(u, [0.5,2.0])

        #test with rmin and rmax set
        p = test_PairPotential.LinPot(types=('1',), params=('m','rmin','rmax'),
                                      default={'m':2,'rmin':0.0,'rmax':1.0})
        u = p.energy(pair=(('1','1')), r=0.5)
        self.assertEqual(u, 1.0)
        u = p.energy(pair=(('1','1')), r=[-0.5,0.0,1.0,1.5])
        np.testing.assert_allclose(u, [0.0,0.0,2.0,2.0])

        #test with shift set
        p = test_PairPotential.LinPot(types=('1',), params=('m','rmax','shift'),
                                      default={'m':2,'rmax':1.0,'shift':True})
        u = p.energy(pair=(('1','1')), r=0.5)
        self.assertEqual(u, -1.0)
        u = p.energy(pair=(('1','1')), r=[1.0,1.5])
        np.testing.assert_allclose(u, [0.0,0.0])

    def test_force(self):
        """Test force method"""
        #test with no cutoffs
        p = test_PairPotential.LinPot(types=('1',), params=('m',),
                                      default={'m':2})
        f = p.force(pair=(('1','1')), r=0.5)
        self.assertEqual(f, -2.0)
        f = p.force(pair=(('1','1')), r=[0.25,0.75])
        np.testing.assert_allclose(f, [-2.0,-2.0])

        #test with rmin set
        p = test_PairPotential.LinPot(types=('1',), params=('m','rmin'),
                                      default={'m':2,'rmin':0.0})
        f = p.force(pair=(('1','1')), r=0.0)
        self.assertEqual(f, -2.0)
        f = p.force(pair=(('1','1')), r=[-0.25,0.75])
        np.testing.assert_allclose(f, [0.0,-2.0])

        #test with rmax set
        p = test_PairPotential.LinPot(types=('1',), params=('m','rmax'),
                                      default={'m':2,'rmax':1.0})
        f = p.force(pair=(('1','1')), r=1.0)
        self.assertEqual(f, -2.0)
        f = p.force(pair=(('1','1')), r=[0.25,1.75])
        np.testing.assert_allclose(f, [-2.0,0.0])

        #test with rmin and rmax set
        p = test_PairPotential.LinPot(types=('1',), params=('m','rmin','rmax'),
                                      default={'m':2,'rmin':0.0,'rmax':1.0})
        f = p.force(pair=(('1','1')), r=0.5)
        self.assertEqual(f, -2.0)
        f = p.force(pair=(('1','1')), r=[-0.5,0.0,1.0,1.5])
        np.testing.assert_allclose(f, [0.0,-2.0,-2.0,0.0])

        #test with shift set
        p = test_PairPotential.LinPot(types=('1',), params=('m','rmax','shift'),
                                      default={'m':2,'rmax':1.0,'shift':True})
        f = p.force(pair=(('1','1')), r=0.5)
        self.assertEqual(f, -2.0)
        f = p.force(pair=(('1','1')), r=[1.0,1.5])
        np.testing.assert_allclose(f, [-2.0,0.0])

    def test_derivative(self):
        """Test derivative method"""
        #test with no cutoffs
        p = test_PairPotential.LinPot(types=('1',), params=('m',),
                                      default={'m':2})
        d = p.derivative(pair=(('1','1')), param='m', r=0.5)
        self.assertEqual(d, 0.5)
        d = p.derivative(pair=(('1','1')), r=[0.25,0.75], param='m')
        np.testing.assert_allclose(d, [0.25,0.75])

        #test with rmin set
        p = test_PairPotential.LinPot(types=('1',), params=('m','rmin'),
                                      default={'m':2,'rmin':0.0})
        d = p.derivative(pair=(('1','1')), param='m', r=0.0)
        self.assertEqual(d, 0.0)
        d = p.derivative(pair=(('1','1')), param='m', r=[-0.25,0.75])
        np.testing.assert_allclose(d, [0.0,0.75])

        #test with rmax set
        p = test_PairPotential.LinPot(types=('1',), params=('m','rmax'),
                                      default={'m':2,'rmax':1.0})
        d = p.derivative(pair=(('1','1')), param='m', r=1.0)
        self.assertEqual(d, 1.0)
        d = p.derivative(pair=(('1','1')), param='m', r=[0.25,1.75])
        np.testing.assert_allclose(d, [0.25,0.0])

        #test with rmin and rmax set
        p = test_PairPotential.LinPot(types=('1',), params=('m','rmin','rmax'),
                                      default={'m':2,'rmin':0.0,'rmax':1.0})
        d = p.derivative(pair=(('1','1')), param='m', r=0.5)
        self.assertEqual(d, 0.5)
        d = p.derivative(pair=(('1','1')), param='m', r=[-0.5,0.0,1.0,1.5])
        np.testing.assert_allclose(d, [0.0,0.0,1.0,0.0])

        #test with shift set
        p = test_PairPotential.LinPot(types=('1',), params=('m','rmax','shift'),
                                      default={'m':2,'rmax':1.0,'shift':True})
        d = p.derivative(pair=(('1','1')), param='m', r=0.5)
        self.assertEqual(d, 0.5)
        d = p.derivative(pair=(('1','1')), param='m', r=[1.0,1.5])
        np.testing.assert_allclose(d, [1.0,0.0])

    def test_iteration(self):
        """Test iteration on coeffs"""
        p = test_PairPotential.LinPot(types=('1','2'), params=('m',))
        for pair in p.coeff:
            p.coeff[pair]['m'] = 2.0
            p.coeff[pair]['rmin'] = 0.0
            p.coeff[pair]['rmax'] = 1.0
        self.assertDictEqual(p.coeff['1','1'].todict(), {'m':2.0, 'rmin':0.0, 'rmax':1.0, 'shift':False})
        self.assertDictEqual(p.coeff['1','2'].todict(), {'m':2.0, 'rmin':0.0, 'rmax':1.0, 'shift':False})
        self.assertDictEqual(p.coeff['2','2'].todict(), {'m':2.0, 'rmin':0.0, 'rmax':1.0, 'shift':False})

    def test_saveload(self):
        """Test saving to and loading from file"""
        temp = tempfile.NamedTemporaryFile()
        p = test_PairPotential.LinPot(types=('1',), params=('m','rmin','rmax'),
                                      default={'m':2,'rmin':0.0,'rmax':1.0})
        x = test_PairPotential.LinPot(types=('1',), params=('m','rmin','rmax','shift'))
        p.save(temp.name)
        x.load(temp.name)

        self.assertEqual(p.coeff['1','1']['m'], x.coeff['1','1']['m'])
        self.assertEqual(p.coeff['1','1']['rmin'], x.coeff['1','1']['rmin'])
        self.assertEqual(p.coeff['1','1']['rmax'], x.coeff['1','1']['rmax'])
        self.assertEqual(p.coeff['1','1']['shift'], x.coeff['1','1']['shift'])

        temp.close()

if __name__ == '__main__':
    unittest.main()