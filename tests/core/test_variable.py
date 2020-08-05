"""Unit tests for core.variable module."""
import unittest

import numpy as np

import relentless

class test_DesignVariable(unittest.TestCase):
    """Unit tests for relentless.DesignVariable."""

    def test_init(self):
        """Test construction with different bounds."""
        #test with no bounds and non-default value of `const`
        v = relentless.DesignVariable(value=1.0, const=True)
        self.assertAlmostEqual(v.value, 1.0)
        self.assertEqual(v.const, True)
        self.assertEqual(v.low, None)
        self.assertEqual(v.high, None)
        self.assertEqual(v.isfree(), True)
        self.assertEqual(v.atlow(), False)
        self.assertEqual(v.athigh(), False)

        #test in between low and high bounds
        v = relentless.DesignVariable(value=1.2, low=0.0, high=2.0)
        self.assertAlmostEqual(v.value, 1.2)
        self.assertAlmostEqual(v.low, 0.0)
        self.assertAlmostEqual(v.high, 2.0)
        self.assertEqual(v.isfree(), True)
        self.assertEqual(v.atlow(), False)
        self.assertEqual(v.athigh(), False)

        #test below low bound
        v = relentless.DesignVariable(value=-1, low=0.5)
        self.assertAlmostEqual(v.value, 0.5)
        self.assertAlmostEqual(v.low, 0.5)
        self.assertEqual(v.high, None)
        self.assertEqual(v.isfree(), False)
        self.assertEqual(v.atlow(), True)
        self.assertEqual(v.athigh(), False)

        #test above high bound
        v = relentless.DesignVariable(value=2.2, high=2.0)
        self.assertAlmostEqual(v.value, 2.0)
        self.assertEqual(v.high, 2.0)
        self.assertEqual(v.low, None)
        self.assertEqual(v.isfree(), False)
        self.assertEqual(v.atlow(), False)
        self.assertEqual(v.athigh(), True)

        #test invalid value initialization
        with self.assertRaises(ValueError):
            v.value = '4'
        #test invalid low initialization
        with self.assertRaises(ValueError):
            v.low = '4'
        #test invalid high initialization
        with self.assertRaises(ValueError):
            v.high = '4'

    def test_clamp(self):
        """Test methods for clamping values with bounds."""
        #construction with only low bound
        v = relentless.DesignVariable(value=0.0, low=2.0)
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
        v = relentless.DesignVariable(value=0.0, low=0.0, high=2.0)
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
        v = relentless.DesignVariable(value=0.0)
        #test free variable
        val,state = v.clamp(1.0)
        self.assertAlmostEqual(val, 1.0)
        self.assertEqual(state, v.State.FREE)

    def test_value(self):
        """Test methods for setting values and checking bounds."""
        #test construction with value between bounds
        v = relentless.DesignVariable(value=0.0, low=-1.0, high=1.0)
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

    def test_bounds(self):
        """Test methods for setting bounds and checking value clamping."""
        #test construction with value between bounds
        v = relentless.DesignVariable(value=0.0, low=-1.0, high=1.0)
        self.assertAlmostEqual(v.value, 0.0)
        self.assertEqual(v.state, v.State.FREE)

        #test changing low bound to above value
        v.low = 0.2
        self.assertAlmostEqual(v.value, 0.2)
        self.assertAlmostEqual(v.low, 0.2)
        self.assertAlmostEqual(v.high, 1.0)
        self.assertEqual(v.state, v.State.LOW)

        #test changing high bound to below value
        v.low = -1.0
        v.high = -0.2
        self.assertAlmostEqual(v.value, -0.2)
        self.assertAlmostEqual(v.low, -1.0)
        self.assertAlmostEqual(v.high, -0.2)
        self.assertEqual(v.state, v.State.HIGH)

        #test changing bounds incorrectly
        with self.assertRaises(ValueError):
            v.low = 0.2
        with self.assertRaises(ValueError):
            v.high = -1.6

class DepVar(relentless.DependentVariable):
    """Mock dependent variable to test relentless.DependentVariable"""
    def __init__(self, *a, **b):
        super().__init__(*a, **b)

    @property
    def value(self):
        v = 0.
        for d in self._attrs:
            v += d.value
        return v

    def _derivative(self, var):
        if self.depends[var].value > 5.0:
            return 0.5
        else:
            return -0.5

class test_DependentVariable(unittest.TestCase):
    """Unit tests for relentless.DependentVariable"""

    def test_init(self):
        """Test creation with data."""
        t = relentless.DesignVariable(value=1.0)
        u = relentless.DesignVariable(value=2.0)
        v = relentless.DesignVariable(value=3.0)

        #test creation with only vardicts
        w = DepVar({'t':t, 'u':u, 'v':v})
        self.assertCountEqual(w._attrs, (t,u,v))
        self.assertDictEqual(w.depends, {'t':t,'u':u,'v':v})
        self.assertAlmostEqual(w.value, 6.0)

        #test creation with only kwvars
        w = DepVar(t=t, u=u, v=v)
        self.assertCountEqual(w._attrs, (t,u,v))
        self.assertDictEqual(w.depends, {'t':t,'u':u,'v':v})
        self.assertAlmostEqual(w.value, 6.0)

        #test creation with vardicts and kwvars
        w = DepVar({'t':t, 'u':u}, v=v)
        self.assertCountEqual(w._attrs, (t,u,v))
        self.assertDictEqual(w.depends, {'t':t,'u':u,'v':v})
        self.assertAlmostEqual(w.value, 6.0)

    def test_derivative(self):
        """Test derivative method."""
        t = relentless.DesignVariable(value=1.0)
        u = relentless.DesignVariable(value=2.0)
        v = relentless.DesignVariable(value=3.0)

        #test derivative with 1 level of dependency
        w = DepVar(t=t, u=u, v=v)
        dw = w.derivative(t)
        self.assertAlmostEqual(dw, -0.5)
        dw = w.derivative(v)
        self.assertAlmostEqual(dw, -0.5)

        x = DepVar(t=t, v=v)
        with self.assertRaises(ValueError):
            dx = x.derivative(u)

        #test derivative with 2 levels of dependency
        y = DepVar(w=w, x=x)
        dy = y.derivative(t)
        self.assertAlmostEqual(dy, 0.0)
        dy = y.derivative(u)
        self.assertAlmostEqual(dy, -0.25)
        dy = y.derivative(v)
        self.assertAlmostEqual(dy, 0.0)

        #test derivative with more complex dependency
        z = DepVar(t=t, w=w, y=y)
        dz = z.derivative(u)
        self.assertAlmostEqual(dz, -0.375)
        dz = z.derivative(v)
        self.assertAlmostEqual(dz, -0.25)
        dz = z.derivative(w)
        self.assertAlmostEqual(dz, 0.75)

        #test derivative with 'multiple' dependency on same object
        q = DepVar(t=t, u=t, v=t)
        dq = q.derivative(t)
        self.assertAlmostEqual(dq, -1.5)

        q = DepVar(t=t, u=t, v=v, w=v)
        dq = q.derivative(t)
        self.assertAlmostEqual(dq, -1.0)
        dq = q.derivative(v)
        self.assertAlmostEqual(dq, -1.0)

        q = DepVar(w=x, x=x)
        dq = q.derivative(t)
        self.assertAlmostEqual(dq, 0.5)

class test_SameAs(unittest.TestCase):
    """Unit tests for relentless.SameAs"""

    def test_init(self):
        """Test creation with data."""
        #test variable dependent on DesignVariable
        v = relentless.DesignVariable(value=1.0)
        w = relentless.SameAs(v)
        self.assertEqual(w.value, 1.0)
        self.assertCountEqual(w._attrs, (v,))
        self.assertDictEqual(w.depends, {'a':v})

        #test variable dependent on DependentVariable
        u = relentless.SameAs(w)
        self.assertEqual(u.value, 1.0)
        self.assertCountEqual(u._attrs, (w,))
        self.assertDictEqual(u.depends, {'a':w})

        #test invalid variable dependence
        with self.assertRaises(TypeError):
            u = relentless.SameAs(1.0)

    def test_value(self):
        """Test setting values."""
        #create "chained" dependent variables
        v = relentless.DesignVariable(value=1.0, low=0.5, high=1.5)
        w = relentless.SameAs(v)
        x = relentless.SameAs(w)

        self.assertEqual(v.value, 1.0)
        self.assertEqual(w.value, 1.0)
        self.assertEqual(x.value, 1.0)

        #change value of v
        v.value = 0.4
        self.assertEqual(v.value, 0.5)
        self.assertEqual(w.value, 0.5)
        self.assertEqual(x.value, 0.5)

    def test_derivative(self):
        """Test _derivative method."""
        v = relentless.DesignVariable(value=1.0)
        w = relentless.SameAs(v)

        #test w.r.t. a
        dw = w._derivative('a')
        self.assertEqual(dw, 1.0)

        #test w.r.t. ~a
        dw = w._derivative('x')
        self.assertEqual(dw, 0.0)

class test_ArithmeticMean(unittest.TestCase):
    """Unit tests for relentless.ArithmeticMean"""

    def test_init(self):
        """Test creation with data."""
        #test variable dependent on 2 DesignVariables
        u = relentless.DesignVariable(value=1.0)
        v = relentless.DesignVariable(value=2.0)
        w = relentless.ArithmeticMean(u, v)
        self.assertAlmostEqual(w.value, 1.5)
        self.assertCountEqual(w._attrs, (u,v))
        self.assertDictEqual(w.depends, {'a':u,'b':v})

        #test variable dependent on 1 DesignVar, 1 DependentVar
        x = relentless.ArithmeticMean(v,w)
        self.assertAlmostEqual(x.value, 1.75)
        self.assertCountEqual(x._attrs, (v,w))
        self.assertDictEqual(x.depends, {'a':v,'b':w})

        #test variable dependent on 2 DependentVariables
        y = relentless.ArithmeticMean(w,x)
        self.assertAlmostEqual(y.value, 1.625)
        self.assertCountEqual(y._attrs, (w,x))
        self.assertDictEqual(y.depends, {'a':w,'b':x})

        #test same-valued dependencies
        z = relentless.DesignVariable(value=1.0)
        w = relentless.ArithmeticMean(u, z)
        self.assertAlmostEqual(w.value, 1.0)
        self.assertCountEqual(w._attrs, (u,z))
        self.assertDictEqual(w.depends, {'a':u,'b':z})

        #test same-object dependencies
        w = relentless.ArithmeticMean(u, u)
        self.assertAlmostEqual(w.value, 1.0)
        self.assertCountEqual(w._attrs, (u,))
        self.assertDictEqual(w.depends, {'a':u,'b':u})

        #test invalid variable dependence
        with self.assertRaises(TypeError):
            z = relentless.ArithmeticMean(1.0, 2.0)

    def test_value(self):
        """Test setting values."""
        #create "chained" dependent variables
        u = relentless.DesignVariable(value=2.0, low=1.5, high=2.5)
        v = relentless.DesignVariable(value=1.0, low=0.5, high=1.5)
        w = relentless.ArithmeticMean(u, v)
        x = relentless.ArithmeticMean(v, w)
        y = relentless.ArithmeticMean(w, x)

        self.assertEqual(u.value, 2.0)
        self.assertEqual(v.value, 1.0)
        self.assertAlmostEqual(w.value, 1.5)
        self.assertAlmostEqual(x.value, 1.25)
        self.assertAlmostEqual(y.value, 1.375)

        #change value of u
        u.value = 2.6
        self.assertEqual(u.value, 2.5)
        self.assertEqual(v.value, 1.0)
        self.assertAlmostEqual(w.value, 1.75)
        self.assertAlmostEqual(x.value, 1.375)
        self.assertAlmostEqual(y.value, 1.5625)

        #change value of v
        v.value = 0.4
        self.assertEqual(u.value, 2.5)
        self.assertEqual(v.value, 0.5)
        self.assertAlmostEqual(w.value, 1.5)
        self.assertAlmostEqual(x.value, 1.0)
        self.assertAlmostEqual(y.value, 1.25)

    def test_derivative(self):
        """Test _derivative method."""
        u = relentless.DesignVariable(value=1.0)
        v = relentless.DesignVariable(value=2.0)
        w = relentless.ArithmeticMean(u, v)

        #test w.r.t. a
        dw = w._derivative('a')
        self.assertEqual(dw, 0.5)

        #test w.r.t. b
        dw = w._derivative('b')
        self.assertEqual(dw, 0.5)

        #test w.r.t. ~a,~b
        dw = w._derivative('c')
        self.assertEqual(dw, 0.0)

class test_GeometricMean(unittest.TestCase):
    """Unit tests for relentless.GeometricMean"""

    def test_init(self):
        """Test creation with data."""
        #test variable dependent on 2 DesignVariables
        u = relentless.DesignVariable(value=1.0)
        v = relentless.DesignVariable(value=2.0)
        w = relentless.GeometricMean(u, v)
        self.assertAlmostEqual(w.value, 1.4142136)
        self.assertCountEqual(w._attrs, (u,v))
        self.assertDictEqual(w.depends, {'a':u,'b':v})

        #test variable dependent on 1 DesignVar, 1 DependentVar
        x = relentless.GeometricMean(v,w)
        self.assertAlmostEqual(x.value, 1.6817928)
        self.assertCountEqual(x._attrs, (v,w))
        self.assertDictEqual(x.depends, {'a':v,'b':w})

        #test variable dependent on 2 DependentVariables
        y = relentless.GeometricMean(w,x)
        self.assertAlmostEqual(y.value, 1.5422108)
        self.assertCountEqual(y._attrs, (w,x))
        self.assertDictEqual(y.depends, {'a':w,'b':x})

        #test same-valued dependencies
        z = relentless.DesignVariable(value=1.0)
        w = relentless.GeometricMean(u, z)
        self.assertAlmostEqual(w.value, 1.0)
        self.assertCountEqual(w._attrs, (u,z))
        self.assertDictEqual(w.depends, {'a':u,'b':z})

        #test same-object dependencies
        w = relentless.GeometricMean(u, u)
        self.assertAlmostEqual(w.value, 1.0)
        self.assertCountEqual(w._attrs, (u,))
        self.assertDictEqual(w.depends, {'a':u,'b':u})

        #test invalid variable dependence
        with self.assertRaises(TypeError):
            z = relentless.GeometricMean(1.0, 2.0)

    def test_value(self):
        """Test setting values."""
        #create "chained" dependent variables
        u = relentless.DesignVariable(value=2.0, low=1.5, high=2.5)
        v = relentless.DesignVariable(value=1.0, low=0.5, high=1.5)
        w = relentless.GeometricMean(u, v)
        x = relentless.GeometricMean(v, w)
        y = relentless.GeometricMean(w, x)

        self.assertEqual(u.value, 2.0)
        self.assertEqual(v.value, 1.0)
        self.assertAlmostEqual(w.value, 1.4142136)
        self.assertAlmostEqual(x.value, 1.1892071)
        self.assertAlmostEqual(y.value, 1.2968396)

        #change value of u
        u.value = 2.6
        self.assertEqual(u.value, 2.5)
        self.assertEqual(v.value, 1.0)
        self.assertAlmostEqual(w.value, 1.5811388)
        self.assertAlmostEqual(x.value, 1.2574334)
        self.assertAlmostEqual(y.value, 1.4100272)

        #change value of v
        v.value = 0.4
        self.assertEqual(u.value, 2.5)
        self.assertEqual(v.value, 0.5)
        self.assertAlmostEqual(w.value, 1.1180340)
        self.assertAlmostEqual(x.value, 0.7476744)
        self.assertAlmostEqual(y.value, 0.9142895)

    def test_derivative(self):
        """Test _derivative method."""
        u = relentless.DesignVariable(value=1.0)
        v = relentless.DesignVariable(value=2.0)
        w = relentless.GeometricMean(u, v)

        #test w.r.t. a
        dw = w._derivative('a')
        self.assertAlmostEqual(dw, 0.7071068)

        #test w.r.t. b
        dw = w._derivative('b')
        self.assertAlmostEqual(dw, 0.3535534)

        #test w.r.t. ~a,~b
        dw = w._derivative('c')
        self.assertEqual(dw, 0.0)

if __name__ == '__main__':
    unittest.main()
