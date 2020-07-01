"""Unit tests for potential module."""
import unittest
import tempfile

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
        default = {'energy':1.0, 'mass':2.0}
        m = potential.CoefficientMatrix(types=('A','B'), params=('energy','mass'),
                                        default={'energy':1.0, 'mass':2.0})
        self.assertEqual(m.types, types)
        self.assertEqual(m.params, params)
        self.assertCountEqual(m.pairs, pairs)

        #test construction with int type parameters
        with self.assertRaises(TypeError):
            m = potential.CoefficientMatrix(types=('A','B'), params=(1,2))

        #test construction with mixed type parameters
        with self.assertRaises(TypeError):
            m = potential.CoefficientMatrix(types=('A','B'), params=('1',2))

    def test_accessors(self):
        """Test get and set methods on pairs"""
        m = potential.CoefficientMatrix(types=('A','B'), params=('energy','mass'))

        #test set and get for each pair type and param
        self.assertEqual(m['A','B']['energy'], None)
        self.assertEqual(m['A','B']['mass'], None)
        self.assertEqual(m['A','A']['energy'], None)
        self.assertEqual(m['A','A']['mass'], None)
        self.assertEqual(m['B','B']['energy'], None)
        self.assertEqual(m['B','B']['mass'], None)

        #test setting all key values at once
        m['A','B'] = {'energy':1.0, 'mass':2.0}
        self.assertEqual(m['A','B']['energy'], 1.0)
        self.assertEqual(m['A','B']['mass'], 2.0)
        self.assertEqual(m['A','A']['energy'], None)
        self.assertEqual(m['A','A']['mass'], None)
        self.assertEqual(m['B','B']['energy'], None)
        self.assertEqual(m['B','B']['mass'], None)

        #test setting key values partially
        m['A','A'] = {'energy':1.0}
        self.assertEqual(m['A','B']['energy'], 1.0)
        self.assertEqual(m['A','B']['mass'], 2.0)
        self.assertEqual(m['A','A']['energy'], 1.0)
        self.assertEqual(m['A','A']['mass'], None)
        self.assertEqual(m['B','B']['energy'], None)
        self.assertEqual(m['B','B']['mass'], None)

        m['A','A'] = {'mass':2.0}
        self.assertEqual(m['A','B']['energy'], 1.0)
        self.assertEqual(m['A','B']['mass'], 2.0)
        self.assertEqual(m['A','A']['energy'], 1.0)
        self.assertEqual(m['A','A']['mass'], 2.0)
        self.assertEqual(m['B','B']['energy'], None)
        self.assertEqual(m['B','B']['mass'], None)

        #test accessing key param values individually
        m['B','B']['energy'] = 1.0
        self.assertEqual(m['A','B']['energy'], 1.0)
        self.assertEqual(m['A','B']['mass'], 2.0)
        self.assertEqual(m['A','A']['energy'], 1.0)
        self.assertEqual(m['A','A']['mass'], 2.0)
        self.assertEqual(m['B','B']['energy'], 1.0)
        self.assertEqual(m['B','B']['mass'], None)

        m['B','B']['mass'] = 2.0
        self.assertEqual(m['A','B']['energy'], 1.0)
        self.assertEqual(m['A','B']['mass'], 2.0)
        self.assertEqual(m['A','A']['energy'], 1.0)
        self.assertEqual(m['A','A']['mass'], 2.0)
        self.assertEqual(m['B','B']['energy'], 1.0)
        self.assertEqual(m['B','B']['mass'], 2.0)

        #test reset and get via iteration
        pairs = (('A','B'), ('B','B'), ('A','A'))
        params = ('energy', 'mass')
        for i in pairs:
            for j in params:
                m[i][j] = 1.5
        self.assertEqual(m['A','B']['energy'], 1.5)
        self.assertEqual(m['A','B']['mass'], 1.5)
        self.assertEqual(m['A','A']['energy'], 1.5)
        self.assertEqual(m['A','A']['mass'], 1.5)
        self.assertEqual(m['B','B']['energy'], 1.5)
        self.assertEqual(m['B','B']['mass'], 1.5)

        #test set and get for initialized default
        m = potential.CoefficientMatrix(types=('A','B'), params=('energy','mass'),
                                        default={'energy':3.0, 'mass':2.5})
        self.assertEqual(m['A','B']['energy'], 3.0)
        self.assertEqual(m['A','B']['mass'], 2.5)
        self.assertEqual(m['A','A']['energy'], 3.0)
        self.assertEqual(m['A','A']['mass'], 2.5)
        self.assertEqual(m['B','B']['energy'], 3.0)
        self.assertEqual(m['B','B']['mass'], 2.5)

        #test reset and get manually
        #test setting all key values at once
        m['A','B'] = {'energy':1.0, 'mass':2.0}
        self.assertEqual(m['A','B']['energy'], 1.0)
        self.assertEqual(m['A','B']['mass'], 2.0)
        self.assertEqual(m['A','A']['energy'], 3.0)
        self.assertEqual(m['A','A']['mass'], 2.5)
        self.assertEqual(m['B','B']['energy'], 3.0)
        self.assertEqual(m['B','B']['mass'], 2.5)

        #test setting key values partially
        m['A','A'] = {'energy':1.0}
        self.assertEqual(m['A','B']['energy'], 1.0)
        self.assertEqual(m['A','B']['mass'], 2.0)
        self.assertEqual(m['A','A']['energy'], 1.0)
        self.assertEqual(m['A','A']['mass'], 2.5)
        self.assertEqual(m['B','B']['energy'], 3.0)
        self.assertEqual(m['B','B']['mass'], 2.5)

        m['A','A'] = {'mass':2.0}
        self.assertEqual(m['A','B']['energy'], 1.0)
        self.assertEqual(m['A','B']['mass'], 2.0)
        self.assertEqual(m['A','A']['energy'], 1.0)
        self.assertEqual(m['A','A']['mass'], 2.0)
        self.assertEqual(m['B','B']['energy'], 3.0)
        self.assertEqual(m['B','B']['mass'], 2.5)

        #test accessing key param values directly
        m['B','B']['energy'] = 1.0
        self.assertEqual(m['A','B']['energy'], 1.0)
        self.assertEqual(m['A','B']['mass'], 2.0)
        self.assertEqual(m['A','A']['energy'], 1.0)
        self.assertEqual(m['A','A']['mass'], 2.0)
        self.assertEqual(m['B','B']['energy'], 1.0)
        self.assertEqual(m['B','B']['mass'], 2.5)

        m['B','B']['mass'] = 2.0
        self.assertEqual(m['A','B']['energy'], 1.0)
        self.assertEqual(m['A','B']['mass'], 2.0)
        self.assertEqual(m['A','A']['energy'], 1.0)
        self.assertEqual(m['A','A']['mass'], 2.0)
        self.assertEqual(m['B','B']['energy'], 1.0)
        self.assertEqual(m['B','B']['mass'], 2.0)

        #test setting for invalid keys and params
        with self.assertRaises(KeyError):
            x = m['A','C']['energy']
        with self.assertRaises(KeyError):
            x = m['A','B']['temperature']

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
                                        default={'energy':1.0, 'mass':2.0})
        self.assertEqual(m.evaluate(('A','B')), {'energy':1.0, 'mass':2.0})
        self.assertEqual(m.evaluate(('A','A')), {'energy':1.0, 'mass':2.0})
        self.assertEqual(m.evaluate(('B','B')), {'energy':1.0, 'mass':2.0})
        self.assertEqual(m.evaluate(('B','A')), m.evaluate(('A','B')))

        #test evaluation with initialized parameter values as Variable types
        m = potential.CoefficientMatrix(types=('A','B'), params=('energy','mass'),
                                        default={'energy':core.Variable(value=1.0), 'mass':core.Variable(value=-1.0,low=0.1)})
        self.assertEqual(m.evaluate(('A','B')), {'energy':1.0, 'mass':0.1})
        self.assertEqual(m.evaluate(('A','A')), {'energy':1.0, 'mass':0.1})
        self.assertEqual(m.evaluate(('B','B')), {'energy':1.0, 'mass':0.1})
        self.assertEqual(m.evaluate(('B','A')), m.evaluate(('A','B')))

        #test evaluation with initialized parameter values as unrecognized types
        m = potential.CoefficientMatrix(types=('A','B'), params=('energy','mass'),
                                        default={'energy':core.Interpolator(x=(-1,0), y=(-2,0)), 'mass':core.Interpolator(x=(0,1), y=(0,2))})
        with self.assertRaises(TypeError):
            x = m.evaluate(('A','B'))

    def test_parse(self):
        """Test saving to and loading from file"""
        temp = tempfile.NamedTemporaryFile()

        #test dumping/re-loading data with scalar parameter values
        m = potential.CoefficientMatrix(types=('A','B'), params=('energy','mass'),
                                        default={'energy':1.0, 'mass':2.0})
        m.save(temp.name)
        x = m.load(temp.name)
        self.assertEqual(m['A','B']['energy'], x['A','B']['energy'])
        self.assertEqual(m['A','B']['mass'], x['A','B']['mass'])
        self.assertEqual(m['A','A']['energy'], x['A','A']['energy'])
        self.assertEqual(m['A','A']['mass'], x['A','A']['mass'])
        self.assertEqual(m['B','B']['energy'], x['B','B']['energy'])
        self.assertEqual(m['B','B']['mass'], x['B','B']['mass'])

        #test dumping/re-loading data with Variable parameter values
        m = potential.CoefficientMatrix(types=('A','B'), params=('energy','mass'),
                                        default={'energy':core.Variable(value=1.0), 'mass':core.Variable(value=-1.0,low=0.1)})
        m.save(temp.name)
        x = m.load(temp.name)
        self.assertEqual(m['A','B']['energy'].value, x['A','B']['energy'])
        self.assertEqual(m['A','B']['mass'].value, x['A','B']['mass'])
        self.assertEqual(m['A','A']['energy'].value, x['A','A']['energy'])
        self.assertEqual(m['A','A']['mass'].value, x['A','A']['mass'])
        self.assertEqual(m['B','B']['energy'].value, x['B','B']['energy'])
        self.assertEqual(m['B','B']['mass'].value, x['B','B']['mass'])

        temp.close()
