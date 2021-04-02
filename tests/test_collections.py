"""Unit tests for _collections module."""
import unittest

import numpy as np

import relentless

class test_FixedKeyDict(unittest.TestCase):
    """Unit tests for relentless._collections.FixedKeyDict."""

    def test_init(self):
        """Test construction with different list keys."""
        keys = ('A','B')
        default = {'A':1.0, 'B':1.0}

        #test construction with tuple input
        d = relentless._collections.FixedKeyDict(keys=('A','B'))
        self.assertEqual(d.keys, keys)
        self.assertEqual([d[k] for k in d.keys], [None, None])

        #test construction with list input
        d = relentless._collections.FixedKeyDict(keys=['A','B'])
        self.assertEqual(d.keys, keys)
        self.assertEqual([d[k] for k in d.keys], [None, None])

        #test construction with defined default input
        d = relentless._collections.FixedKeyDict(keys=('A','B'), default=1.0)
        self.assertEqual(d.keys, keys)
        self.assertEqual([d[k] for k in d.keys], [1.0, 1.0])

        #test construction with single-key tuple input
        keys = ('A',)
        d = relentless._collections.FixedKeyDict(keys=('A',))
        self.assertEqual(d.keys, keys)
        self.assertEqual([d[k] for k in d.keys], [None])

    def test_accessors(self):
        """Test get and set methods on keys."""
        d = relentless._collections.FixedKeyDict(keys=('A','B'))

        #test setting and getting values
        d['A'] = 1.0
        self.assertEqual([d[k] for k in d.keys], [1.0, None])
        d['B'] = 1.0
        self.assertEqual([d[k] for k in d.keys], [1.0, 1.0])

        #test re-setting and getting values
        d['A'] = 2.0
        self.assertEqual([d[k] for k in d.keys], [2.0, 1.0])
        d['B'] = 1.5
        self.assertEqual([d[k] for k in d.keys], [2.0, 1.5])

        #test getting invalid key
        with self.assertRaises(KeyError):
            x = d['C']

    def test_update(self):
        """Test update method to get and set keys."""
        d = relentless._collections.FixedKeyDict(keys=('A','B'))

        self.assertEqual([d[k] for k in d.keys], [None, None])

        #test updating both keys
        d.update({'A':1.0, 'B':2.0})  #using dict
        self.assertEqual([d[k] for k in d.keys], [1.0, 2.0])

        d.update(A=1.5, B=2.5)  #using kwargs
        self.assertEqual([d[k] for k in d.keys], [1.5, 2.5])

        #test updating only one key at a time
        d.update({'A':1.1})   #using dict
        self.assertEqual([d[k] for k in d.keys], [1.1, 2.5])

        d.update(B=2.2)   #using kwargs
        self.assertEqual([d[k] for k in d.keys], [1.1, 2.2])

        #test using *args length > 1
        with self.assertRaises(TypeError):
            d.update({'A':3.0}, {'B':4.0})

        #test using both *args and **kwargs
        d.update({'A':3.0, 'B':2.0}, B=2.2)
        self.assertEqual([d[k] for k in d.keys], [3.0, 2.2])

        #test using invalid kwarg
        with self.assertRaises(KeyError):
            d.update(C=2.5)

    def test_clear(self):
        """Test clear method to reset keys to default."""
        #test clear with no default set
        d = relentless._collections.FixedKeyDict(keys=('A','B'))
        self.assertEqual([d[k] for k in d.keys], [None, None])
        d.update(A=2, B=3)
        self.assertEqual([d[k] for k in d.keys], [2.0, 3.0])
        d.clear()
        self.assertEqual([d[k] for k in d.keys], [None, None])

        #test clear with set default
        d = relentless._collections.FixedKeyDict(keys=('A','B'), default=1.0)
        self.assertEqual([d[k] for k in d.keys], [1.0, 1.0])
        d.update(A=2, B=3)
        self.assertEqual([d[k] for k in d.keys], [2.0, 3.0])
        d.clear()
        self.assertEqual([d[k] for k in d.keys], [1.0, 1.0])

    def test_iteration(self):
        """Test iteration on the dictionary."""
        d = relentless._collections.FixedKeyDict(keys=('A','B'))

        #test iteration for setting values
        for k in d:
            d[k] = 1.0
        self.assertEqual([d[k] for k in d.keys], [1.0, 1.0])

        #test manual re-setting of values
        d['A'] = 2.0
        self.assertEqual([d[k] for k in d.keys], [2.0, 1.0])
        d['B'] = 1.5
        self.assertEqual([d[k] for k in d.keys], [2.0, 1.5])

        #test iteration for re-setting values
        for k in d:
            d[k] = 3.0
        self.assertEqual([d[k] for k in d.keys], [3.0, 3.0])

    def test_copy(self):
        """Test copying custom dict to standard dict."""
        d = relentless._collections.FixedKeyDict(keys=('A','B'))

        #test copying for empty dict
        dict_var = {'A':None, 'B':None}
        self.assertEqual(d.todict(), dict_var)

        #test copying for partially filled dict
        dict_var = {'A':None, 'B':1.0}
        d['B'] = 1.0
        self.assertEqual(d.todict(), dict_var)

        #test copying for full dict
        dict_var = {'A':1.0, 'B':1.0}
        d['A'] = 1.0
        self.assertEqual(d.todict(), dict_var)

class test_PairMatrix(unittest.TestCase):
    """Unit tests for relentless._collections.PairMatrix."""

    def test_init(self):
        """Test construction with different list types."""
        types = ('A','B')
        pairs  = (('A','B'), ('B','B'), ('A','A'))

        #test construction with tuple input
        m = relentless._collections.PairMatrix(types=('A','B'))
        self.assertEqual(m.types, types)
        self.assertCountEqual(m.pairs, pairs)

        #test construction with list input
        m = relentless._collections.PairMatrix(types=['A','B'])
        self.assertEqual(m.types, types)
        self.assertCountEqual(m.pairs, pairs)

        types = ('A',)
        pairs = (('A','A'),)

        #test construction with single type tuple
        m = relentless._collections.PairMatrix(types=('A',))
        self.assertEqual(m.types, types)
        self.assertCountEqual(m.pairs, pairs)

        #test construction with int type input
        with self.assertRaises(TypeError):
            m = relentless._collections.PairMatrix(types=(1,2))

        #test construction with mixed type input
        with self.assertRaises(TypeError):
            m = relentless._collections.PairMatrix(types=('1',2))

    def test_accessors(self):
        """Test get and set methods on pairs."""
        m = relentless._collections.PairMatrix(types=('A','B'))

        #test set and get for each pair type
        m['A','A']['energy'] = 1.0
        self.assertEqual(m['A','A']['energy'], 1.0)
        self.assertEqual(m['A','B'], {})
        self.assertEqual(m['B','B'], {})

        m['A','B']['energy'] = -1.0
        self.assertEqual(m['A','A']['energy'], 1.0)
        self.assertEqual(m['A','B']['energy'], -1.0)
        self.assertEqual(m['B','B'], {})

        m['B','B']['energy'] = 1.0
        self.assertEqual(m['A','A']['energy'], 1.0)
        self.assertEqual(m['A','B']['energy'], -1.0)
        self.assertEqual(m['B','B']['energy'], 1.0)

        #test key order equality
        self.assertEqual(m['A','B'], m['B','A'])

        #test re-set and get
        m['A','A']['energy'] = 2.0
        self.assertEqual(m['A','A']['energy'], 2.0)
        self.assertEqual(m['A','B']['energy'], -1.0)
        self.assertEqual(m['B','B']['energy'], 1.0)

        m['A','B']['energy'] = -1.5
        self.assertEqual(m['A','A']['energy'], 2.0)
        self.assertEqual(m['A','B']['energy'], -1.5)
        self.assertEqual(m['B','B']['energy'], 1.0)

        m['B','B']['energy'] = 0.0
        self.assertEqual(m['A','A']['energy'], 2.0)
        self.assertEqual(m['A','B']['energy'], -1.5)
        self.assertEqual(m['B','B']['energy'], 0.0)

        #test setting multiple parameters and get
        m['A','A']['mass'] = 1.0
        self.assertEqual(m['A','A']['mass'], 1.0)
        self.assertEqual(m['A','A']['energy'], 2.0)
        self.assertEqual(m['A','A'], {'energy':2.0, 'mass':1.0})

        m['A','B']['mass'] = 3.0
        self.assertEqual(m['A','B']['mass'], 3.0)
        self.assertEqual(m['A','B']['energy'], -1.5)
        self.assertEqual(m['A','B'], {'energy':-1.5, 'mass':3.0})

        m['B','B']['mass'] = 5.0
        self.assertEqual(m['B','B']['mass'], 5.0)
        self.assertEqual(m['B','B']['energy'], 0.0)
        self.assertEqual(m['B','B'], {'energy':0.0, 'mass':5.0})

        #test setting paramters for invalid keys
        with self.assertRaises(KeyError):
            x = m['C','C']
        with self.assertRaises(KeyError):
            x = m['A','C']

    def test_iteration(self):
        """Test iteration on the matrix."""
        m = relentless._collections.PairMatrix(types=('A','B'))

        #test iteration for initialization
        for pair in m:
            m[pair]['mass'] = 2.0
            m[pair]['energy'] = 1.0
        self.assertEqual(m['A','B'], {'energy':1.0, 'mass':2.0})
        self.assertEqual(m['A','A'], {'energy':1.0, 'mass':2.0})
        self.assertEqual(m['B','B'], {'energy':1.0, 'mass':2.0})

        #test resetting values manually
        m['A','B']['mass'] = 2.5
        m['A','A']['energy'] = 1.5
        self.assertEqual(m['A','B'], {'energy':1.0, 'mass':2.5})
        self.assertEqual(m['A','A'], {'energy':1.5, 'mass':2.0})
        self.assertEqual(m['B','B'], {'energy':1.0, 'mass':2.0})

        #test re-iteration for setting values
        for pair in m:
            m[pair]['energy'] = 3.0
        self.assertEqual(m['A','B'], {'energy':3.0, 'mass':2.5})
        self.assertEqual(m['A','A'], {'energy':3.0, 'mass':2.0})
        self.assertEqual(m['B','B'], {'energy':3.0, 'mass':2.0})

class test_KeyedArray(unittest.TestCase):
    """Unit tests for relentless._collections.KeyedArray."""

    def test_init(self):
        """Test construction with data."""
        k = relentless._collections.KeyedArray(keys=('A','B'))
        self.assertDictEqual(k.todict(), {'A':None, 'B':None})

        k = relentless._collections.KeyedArray(keys=('A','B'), default=2.0)
        self.assertDictEqual(k.todict(), {'A':2.0, 'B':2.0})

        #invalid key
        with self.assertRaises(KeyError):
            x = k['C']

    def test_arithmetic_ops(self):
        """Test arithmetic operations."""
        k1 = relentless._collections.KeyedArray(keys=('A','B'))
        k1.update({'A':1.0, 'B':2.0})
        k2 = relentless._collections.KeyedArray(keys=('A','B'))
        k2.update({'A':2.0, 'B':3.0})
        k3 = relentless._collections.KeyedArray(keys=('A','B','C'))
        k3.update({'A':3.0, 'B':4.0, 'C':5.0})

        #addition
        k4 = k1 + k2
        self.assertDictEqual(k4.todict(), {'A':3.0, 'B':5.0})
        k4 += k2
        self.assertDictEqual(k4.todict(), {'A':5.0, 'B':8.0})
        k4 = k1 + 1
        self.assertDictEqual(k4.todict(), {'A':2.0, 'B':3.0})
        k4 = 2 + k1
        self.assertDictEqual(k4.todict(), {'A':3.0, 'B':4.0})
        k4 += 1
        self.assertDictEqual(k4.todict(), {'A':4.0, 'B':5.0})
        with self.assertRaises(KeyError):
            k4 = k1 + k3
        with self.assertRaises(KeyError):
            k3 += k2

        #subtraction
        k4 = k1 - k2
        self.assertDictEqual(k4.todict(), {'A':-1.0, 'B':-1.0})
        k4 -= k2
        self.assertDictEqual(k4.todict(), {'A':-3.0, 'B':-4.0})
        k4 = k1 - 1
        self.assertDictEqual(k4.todict(), {'A':0.0, 'B':1.0})
        k4 = 2 - k1
        self.assertDictEqual(k4.todict(), {'A':1.0, 'B':0.0})
        k4 -= 1
        self.assertDictEqual(k4.todict(), {'A':0.0, 'B':-1.0})
        with self.assertRaises(KeyError):
            k4 = k1 - k3
        with self.assertRaises(KeyError):
            k3 -= k2

        #multiplication
        k4 = k1*k2
        self.assertDictEqual(k4.todict(), {'A':2.0, 'B':6.0})
        k4 *= k2
        self.assertDictEqual(k4.todict(), {'A':4.0, 'B':18.0})
        k4 = 3*k1
        self.assertDictEqual(k4.todict(), {'A':3.0, 'B':6.0})
        k4 = k2*3
        self.assertDictEqual(k4.todict(), {'A':6.0, 'B':9.0})
        k4 *= 3
        self.assertDictEqual(k4.todict(), {'A':18.0, 'B':27.0})
        with self.assertRaises(KeyError):
            k4 = k1*k3

        #division
        k4 = k1/k2
        self.assertDictEqual(k4.todict(), {'A':0.5, 'B':0.6666666666666666})
        k4 /= k2
        self.assertDictEqual(k4.todict(), {'A':0.25, 'B':0.2222222222222222})
        k4 = 2/k2
        self.assertDictEqual(k4.todict(), {'A':1.0, 'B':0.6666666666666666})
        k4 = k2/2
        self.assertDictEqual(k4.todict(), {'A':1.0, 'B':1.5})
        k4 /= 2
        self.assertDictEqual(k4.todict(), {'A':0.5, 'B':0.75})
        with self.assertRaises(KeyError):
            k4 = k1/k3

        #exponentiation
        k4 = k1**k2
        self.assertDictEqual(k4.todict(), {'A':1.0, 'B':8.0})
        k4 = k2**2
        self.assertDictEqual(k4.todict(), {'A':4.0, 'B':9.0})

        #negation
        k4 = -k1
        self.assertDictEqual(k4.todict(), {'A':-1.0, 'B':-2.0})

    def test_vector_ops(self):
        """Test vector operations."""
        k1 = relentless._collections.KeyedArray(keys=('A','B'))
        k1.update({'A':1.0, 'B':2.0})
        k2 = relentless._collections.KeyedArray(keys=('A','B'))
        k2.update({'A':2.0, 'B':3.0})
        k3 = relentless._collections.KeyedArray(keys=('A','B','C'))
        k3.update({'A':3.0, 'B':4.0, 'C':5.0})

        #norm
        self.assertAlmostEqual(k1.norm(), np.sqrt(5))
        self.assertAlmostEqual(k3.norm(), np.sqrt(50))

        #dot product
        self.assertAlmostEqual(k1.dot(k2), 8.0)
        self.assertAlmostEqual(k1.dot(k2), k2.dot(k1))
        with self.assertRaises(KeyError):
            k4 = k2.dot(k3)

if __name__ == '__main__':
    unittest.main()
