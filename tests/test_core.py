"""Unit tests for core module."""
import unittest
import numpy as np

import relentless

class test_Interpolator(unittest.TestCase):
    """Unit tests for core.Interpolator."""

    def test_init(self):
        """Test creation from data."""
        raise NotImplementedError()

    def test_call(self):
        """Test calls, both scalar and array."""
        raise NotImplementedError()

    def test_extrap(self):
        """Test extrapolation calls."""
        raise NotImplementedError()

class test_PairMatrix(unittest.TestCase):
    """Unit tests for core.PairMatrix."""

    def test_init(self):
        """Test construction with different list types."""
        raise NotImplementedError()

    def test_accessors(self):
        """Test get and set methods on pairs."""
        raise NotImplementedError()

    def test_iteration(self):
        """Test iteration on the matrix."""
        raise NotImplementedError()

class test_TypeDict(unittest.TestCase):
    """Unit tests for core.TypeDict."""

    def test_init(self):
        """Test construction with different list types."""

        types = ('A','B')

        #test construction with tuple input
        d = relentless.core.TypeDict(types=('A','B'))
        self.assertEqual(d.types, types)

        #test construction with list input
        d = relentless.core.TypeDict(types=['A','B'])
        self.assertEqual(d.types, types)

        #test construction with numpy array input
        d = relentless.core.TypeDict(types=np.array(['A','B']))
        self.assertEqual(d.types, types)

        types = ('A',)

        #test construction with single-type tuple input
        d = relentless.core.TypeDict(types=('A',))
        self.assertEqual(d.types, types)

        #test construction with int type input
        with self.assertRaises(TypeError):
            d = relentless.core.TypeDict(types=(1,2))

        #test construction with mixed type input
        with self.assertRaises(TypeError):
            d = relentless.core.TypeDict(types=('1',2))

    def test_accessors(self):
        """Test get and set methods on types."""

        d = relentless.core.TypeDict(types=('A','B'))

        #test setting and getting values
        d['A'] = 1.0
        self.assertEqual(d['A'], 1.0)
        self.assertEqual(d['B'], None)

        d['B'] = 1.0
        self.assertEqual(d['A'], 1.0)
        self.assertEqual(d['B'], 1.0)

        #test re-setting and getting values
        d['A'] = 2.0
        self.assertEqual(d['A'], 2.0)
        self.assertEqual(d['B'], 1.0)

        d['B'] = 1.5
        self.assertEqual(d['A'], 2.0)
        self.assertEqual(d['B'], 1.5)

        #test getting invalid key
        with self.assertRaises(KeyError):
            x = d['C']

    def test_iteration(self):
        """Test iteration on the dictionary."""

        d = relentless.core.TypeDict(types=('A','B'))

        #test iteration for setting values
        for t in d:
            d[t] = 1.0
        self.assertEqual(d['A'], 1.0)
        self.assertEqual(d['B'], 1.0)

        #test manual re-setting of values
        d['A'] = 2.0
        self.assertEqual(d['A'], 2.0)
        self.assertEqual(d['B'], 1.0)

        d['B'] = 1.5
        self.assertEqual(d['A'], 2.0)
        self.assertEqual(d['B'], 1.5)

        #test iteration for re-setting values
        for t in d:
            d[t] = 3.0
        self.assertEqual(d['A'], 3.0)
        self.assertEqual(d['B'], 3.0)

    def test_copy(self):
        """Test copying custom dict to standard dict."""

        d = relentless.core.TypeDict(types=('A','B'))

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
