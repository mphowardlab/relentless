"""Unit tests for core module."""
import unittest
import numpy as np

import relentless

class test_Interpolator(unittest.TestCase):
    """Unit tests for core.Interpolator."""

    def test_init(self):
        """Test creation from data."""

        #test construction with tuple input
        f = relentless.core.Interpolator(x=(-1,0,1), y=(-2,0,2))
        self.assertEqual(f.domain, (-1,1))

        #test construction with list input
        f = relentless.core.Interpolator(x=[-1,0,1], y=[-2,0,2])
        self.assertEqual(f.domain, (-1,1))

        #test construction with numpy array input
        f = relentless.core.Interpolator(x=np.array([-1,0,1]),
                                         y=np.array([-2,0,2]))
        self.assertEqual(f.domain, (-1,1))

        #test construction with mixed input
        f = relentless.core.Interpolator(x=[-1,0,1], y=(-2,0,2))
        self.assertEqual(f.domain, (-1,1))

        #test construction with scalar input
        with self.assertRaises(ValueError):
            f = relentless.core.Interpolator(x=1, y=2)

        #test construction with 2d-array input
        with self.assertRaises(ValueError):
            f = relentless.core.Interpolator(x=np.array([[-1,0,1], [-2,2,4]]),
                                             y=np.array([[-1,0,1], [-2,2,4]]))

        #test construction with x and y having different lengths
        with self.assertRaises(ValueError):
            f = relentless.core.Interpolator(x=[-1,0], y=[-2,0,2])

        #test construction with non-strictly-increasing domain
        with self.assertRaises(ValueError):
            f = relentless.core.Interpolator(x=(0,1,-1), y=(0,2,-2))

    def test_call(self):
        """Test calls, both scalar and array."""
        f = relentless.core.Interpolator(x=(-1,0,1), y=(-2,0,2))

        #test scalar call
        self.assertAlmostEqual(f(-0.5), -1.0)
        self.assertAlmostEqual(f(0.5), 1.0)

        #test array call
        np.testing.assert_allclose(f([-0.5,0.5]), [-1.0,1.0])

    def test_extrap(self):
        """Test extrapolation calls."""
        f = relentless.core.Interpolator(x=(-1,0,1), y=(-2,0,2))

        #test extrap below lo
        self.assertAlmostEqual(f(-2), -2.0)

        #test extrap above hi
        self.assertAlmostEqual(f(2), 2.0)

        #test extrap below low and above hi
        np.testing.assert_allclose(f([-2,2]), [-2.0,2.0])

        #test combined extrapolation and interpolation
        np.testing.assert_allclose(f([-2,0.5,2]), [-2.0,1.0,2.0])

class test_PairMatrix(unittest.TestCase):
    """Unit tests for core.PairMatrix."""

    def test_init(self):
        """Test construction with different list types."""

        types = ('A','B')
        pairs  = (('A','B'), ('B','B'), ('A','A'))

        #test construction with tuple input
        m = relentless.core.PairMatrix(types=('A','B'))
        self.assertEqual(m.types, types)
        self.assertCountEqual(m.pairs, pairs)

        #test construction with list input
        m = relentless.core.PairMatrix(types=['A','B'])
        self.assertEqual(m.types, types)
        self.assertCountEqual(m.pairs, pairs)

        types = ('A',)
        pairs = (('A','A'),)

        #test construction with single type tuple
        m = relentless.core.PairMatrix(types=('A',))
        self.assertEqual(m.types, types)
        self.assertCountEqual(m.pairs, pairs)

        #test construction with int type input
        with self.assertRaises(TypeError):
            m = relentless.core.PairMatrix(types=(1,2))

        #test construction with mixed type input
        with self.assertRaises(TypeError):
            m = relentless.core.PairMatrix(types=('1',2))

    def test_accessors(self):
        """Test get and set methods on pairs."""

        m = relentless.core.PairMatrix(types=('A','B'))

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
        m = relentless.core.PairMatrix(types=('A','B'))

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

class test_TypeDict(unittest.TestCase):
    """Unit tests for core.TypeDict."""

    def test_init(self):
        """Test construction with different list types."""
        raise NotImplementedError()

    def test_accessors(self):
        """Test get and set methods on types."""
        raise NotImplementedError()

    def test_iteration(self):
        """Test iteration on the dictionary."""
        raise NotImplementedError()

    def test_copy(self):
        """Test copying custom dict to standard dict."""
        raise NotImplementedError()

class test_Variable(unittest.TestCase):
    """Unit tests for core.Variable."""

    def test_init(self):
        """Test construction with different bounds."""

        #test with no bounds and non-default value of `const`
        v = relentless.core.Variable(value=1.0, const=True)
        self.assertAlmostEqual(v.value, 1.0)
        self.assertEqual(v.const, True)
        self.assertEqual(v.low, None)
        self.assertEqual(v.high, None)
        self.assertEqual(v.isfree(), True)
        self.assertEqual(v.atlow(), False)
        self.assertEqual(v.athigh(), False)

        #test in between low and high bounds
        v = relentless.core.Variable(value=1.2, low=0.0, high=2.0)
        self.assertAlmostEqual(v.value, 1.2)
        self.assertAlmostEqual(v.low, 0.0)
        self.assertAlmostEqual(v.high, 2.0)
        self.assertEqual(v.isfree(), True)
        self.assertEqual(v.atlow(), False)
        self.assertEqual(v.athigh(), False)

        #test below low bound
        v = relentless.core.Variable(value=-1, low=0.5)
        self.assertAlmostEqual(v.value, 0.5)
        self.assertAlmostEqual(v.low, 0.5)
        self.assertEqual(v.high, None)
        self.assertEqual(v.isfree(), False)
        self.assertEqual(v.atlow(), True)
        self.assertEqual(v.athigh(), False)

        #test above high bound
        v = relentless.core.Variable(value=2.2, high=2.0)
        self.assertAlmostEqual(v.value, 2.0)
        self.assertEqual(v.high, 2.0)
        self.assertEqual(v.low, None)
        self.assertEqual(v.isfree(), False)
        self.assertEqual(v.atlow(), False)
        self.assertEqual(v.athigh(), True)

        #test invalid value initialization
        with self.assertRaises(ValueError):
            v = relentless.core.Variable(value='4')

    def test_clamp(self):
        """Test methods for clamping values with bounds."""

        #construction with only low bound
        v = relentless.core.Variable(value=0.0, low=2.0)
        #test below low
        val,state = v.clamp(1.0)
        self.assertAlmostEqual(val, 2.0)
        self.assertEqual(state, v.State.LOW)
        #test at low
        val,state = v.clamp(2.0)
        self.assertAlmostEqual(val, 2.0)
        self.assertEqual(state, v.State.LOW)
        #test above low
        val,state = v.clamp(3.0)
        self.assertAlmostEqual(val, 3.0)
        self.assertEqual(state, v.State.FREE)

        #construction with low and high bounds
        v = relentless.core.Variable(value=0.0, low=0.0, high=2.0)
        #test below low
        val,state = v.clamp(-1.0)
        self.assertAlmostEqual(val, 0.0)
        self.assertEqual(state, v.State.LOW)
        #test between bounds
        val,state = v.clamp(1.0)
        self.assertAlmostEqual(val, 1.0)
        self.assertEqual(state, v.State.FREE)
        #test above high
        val,state = v.clamp(2.5)
        self.assertAlmostEqual(val, 2.0)
        self.assertEqual(state, v.State.HIGH)

        #construction with no bounds
        v = relentless.core.Variable(value=0.0)
        #test free variable
        val,state = v.clamp(1.0)
        self.assertAlmostEqual(val, 1.0)
        self.assertEqual(state, v.State.FREE)

    def test_value(self):
        """Test methods for setting values and checking bounds."""

        #test construction with value between bounds
        v = relentless.core.Variable(value=0.0, low=-1.0, high=1.0)
        self.assertAlmostEqual(v.value, 0.0)
        self.assertEqual(v.state, v.State.FREE)

        #test below low
        v.value = -1.5
        self.assertAlmostEqual(v.value, -1.0)
        self.assertEqual(v.state, v.State.LOW)

        #test above high
        v.value = 3
        self.assertAlmostEqual(v.value, 1.0)
        self.assertEqual(v.state, v.State.HIGH)

        #test invalid value
        with self.assertRaises(ValueError):
            v.value = '0'

if __name__ == '__main__':
    unittest.main()
