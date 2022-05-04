"""Unit tests for core.variable module."""
import unittest

import numpy

import relentless

class test_Variable(unittest.TestCase):
    """Unit tests for relentless.variable.Variable."""

    def test_operations(self):
        """Test arithmetic operations on variables."""
        #create dependent variables
        v = relentless.variable.IndependentVariable(value=2.0)
        w = relentless.variable.IndependentVariable(value=4.0)

        #addition
        x = v+w
        self.assertAlmostEqual(x.value, 6.0)
        x = v+2.0
        self.assertAlmostEqual(x.value, 4.0)
        x = 2.0+w
        self.assertAlmostEqual(x.value, 6.0)

        #subtraction
        x = v-w
        self.assertAlmostEqual(x.value, -2.0)
        x = v-1.0
        self.assertAlmostEqual(x.value, 1.0)
        x = 1.0-w
        self.assertAlmostEqual(x.value, -3.0)

        #multiplication
        x = v*w
        self.assertAlmostEqual(x.value, 8.0)
        x = v*1.5
        self.assertAlmostEqual(x.value, 3.0)
        x = 1.5*w
        self.assertAlmostEqual(x.value, 6.0)

        #division
        x = w/v
        self.assertAlmostEqual(x.value, 2.0)
        x = v/2.0
        self.assertAlmostEqual(x.value, 1.0)
        x = 3.0/w
        self.assertAlmostEqual(x.value, 0.75)

        #exponentiation
        x = w**v
        self.assertAlmostEqual(x.value, 16.0)
        x = v**3.0
        self.assertAlmostEqual(x.value, 8.0)

        #negation
        x = -w
        self.assertAlmostEqual(x.value, -4.0)

        #string conversion
        self.assertEqual(str(x), '-4.0')

class test_IndependentVariable(unittest.TestCase):
    """Unit tests for relentless.variable.IndependentVariable."""

    def test_init(self):
        """Test construction with data."""
        #test scalar values
        v = relentless.variable.IndependentVariable(1.0)
        self.assertEqual(v.value, 1.0)

        v = relentless.variable.IndependentVariable(1)
        self.assertEqual(v.value, 1)

        with self.assertRaises(TypeError):
            v = relentless.variable.IndependentVariable('1.0')

    def test_value(self):
        """Test value setting."""
        v = relentless.variable.IndependentVariable(1.0)
        self.assertEqual(v.value, 1.0)

        v.value = 2
        self.assertEqual(v.value, 2)

    def test_ioperations(self):
        """Test in-place arithmetic operations on variables."""
        #create dependent variables
        v = relentless.variable.IndependentVariable(value=2.0)
        w = relentless.variable.IndependentVariable(value=4.0)

        #addition
        v += 2.0
        self.assertAlmostEqual(v.value, 4.0)
        with self.assertRaises(TypeError):
            v += w

        #subtraction
        w -= 2.0
        self.assertAlmostEqual(w.value, 2.0)
        with self.assertRaises(TypeError):
            v -= w

        #multiplication
        w *= 3.0
        self.assertAlmostEqual(w.value, 6.0)
        with self.assertRaises(TypeError):
            v *= w

        #division
        v /= 4.0
        self.assertAlmostEqual(v.value, 1.0)
        with self.assertRaises(TypeError):
            v /= w

class test_DesignVariable(unittest.TestCase):
    """Unit tests for relentless.variable.DesignVariable."""

    def test_init(self):
        """Test construction with different bounds."""
        #test with no bounds
        v = relentless.variable.DesignVariable(value=1.0)
        self.assertAlmostEqual(v.value, 1.0)
        self.assertEqual(v.low, None)
        self.assertEqual(v.high, None)
        self.assertEqual(v.isfree(), True)
        self.assertEqual(v.atlow(), False)
        self.assertEqual(v.athigh(), False)

        #test in between low and high bounds
        v = relentless.variable.DesignVariable(value=1.2, low=0.0, high=2.0)
        self.assertAlmostEqual(v.value, 1.2)
        self.assertAlmostEqual(v.low, 0.0)
        self.assertAlmostEqual(v.high, 2.0)
        self.assertEqual(v.isfree(), True)
        self.assertEqual(v.atlow(), False)
        self.assertEqual(v.athigh(), False)

        #test below low bound
        v = relentless.variable.DesignVariable(value=-1, low=0.5)
        self.assertAlmostEqual(v.value, 0.5)
        self.assertAlmostEqual(v.low, 0.5)
        self.assertEqual(v.high, None)
        self.assertEqual(v.isfree(), False)
        self.assertEqual(v.atlow(), True)
        self.assertEqual(v.athigh(), False)

        #test above high bound
        v = relentless.variable.DesignVariable(value=2.2, high=2.0)
        self.assertAlmostEqual(v.value, 2.0)
        self.assertEqual(v.high, 2.0)
        self.assertEqual(v.low, None)
        self.assertEqual(v.isfree(), False)
        self.assertEqual(v.atlow(), False)
        self.assertEqual(v.athigh(), True)

        #test setting bounds after construction
        v = relentless.variable.DesignVariable(value=-1.0)
        self.assertAlmostEqual(v.value, -1.0)
        self.assertEqual(v.high, None)
        self.assertEqual(v.low, None)
        #only low
        v.low = -1.5
        self.assertAlmostEqual(v.value, -1.0)
        self.assertEqual(v.high, None)
        self.assertEqual(v.low, -1.5)
        #only high
        v.low = None
        v.high = 1.5
        self.assertAlmostEqual(v.value, -1.0)
        self.assertEqual(v.high, 1.5)
        self.assertEqual(v.low, None)
        #both
        v.low = -1.5
        self.assertAlmostEqual(v.value, -1.0)
        self.assertEqual(v.high, 1.5)
        self.assertEqual(v.low, -1.5)

        with self.assertRaises(TypeError):
            v.value = '4'
        #test invalid low initialization
        with self.assertRaises(TypeError):
            v.low = '4'
        #test invalid high initialization
        with self.assertRaises(TypeError):
            v.high = '4'

    def test_clamp(self):
        """Test methods for clamping values with bounds."""
        #construction with only low bound
        v = relentless.variable.DesignVariable(value=0.0, low=2.0)
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
        v = relentless.variable.DesignVariable(value=0.0, low=0.0, high=2.0)
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
        v = relentless.variable.DesignVariable(value=0.0)
        #test free variable
        val,state = v.clamp(1.0)
        self.assertAlmostEqual(val, 1.0)
        self.assertEqual(state, v.State.FREE)

    def test_value(self):
        """Test methods for setting values and checking bounds."""
        #test construction with value between bounds
        v = relentless.variable.DesignVariable(value=0.0, low=-1.0, high=1.0)
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
        with self.assertRaises(TypeError):
            v.value = '0'

    def test_bounds(self):
        """Test methods for setting bounds and checking value clamping."""
        #test construction with value between bounds
        v = relentless.variable.DesignVariable(value=0.0, low=-1.0, high=1.0)
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

class DepVar(relentless.variable.DependentVariable):
    """Mock dependent variable to test relentless.variable.DependentVariable"""
    def __init__(self, *a, **b):
        super().__init__(*a, **b)

    def compute(self, **kwargs):
        return numpy.sum([v for v in kwargs.values()])

    def compute_derivative(self, param, **kwargs):
        #Note: this method doesn't calculate the actual derivatives;
        #it is merely used for testing the chain rule calculations
        #with various variable dependencies
        if param in self.params:
            if kwargs[param] > 5.0:
                return 0.5
            else:
                return -0.5
        else:
            raise ValueError('Unknown parameter')

class test_DependentVariable(unittest.TestCase):
    """Unit tests for relentless.variable.DependentVariable"""

    def test_init(self):
        """Test creation with data."""
        t = relentless.variable.DesignVariable(value=1.0)
        u = relentless.variable.DesignVariable(value=2.0)
        v = relentless.variable.DesignVariable(value=3.0)

        #test creation with only vardicts
        w = DepVar({'t':t, 'u':u, 'v':v})
        self.assertAlmostEqual(w.value, 6.0)

        #test creation with only kwvars
        w = DepVar(t=t, u=u, v=v)
        self.assertAlmostEqual(w.value, 6.0)

        #test creation with vardicts and kwvars
        w = DepVar({'t':t, 'u':u}, v=v)
        self.assertAlmostEqual(w.value, 6.0)

        #test creation with repeated attributes
        w = DepVar({'t':t, 'u':u}, u=v)
        self.assertAlmostEqual(w.value, 4.0)

        #test creation with multiple vardicts
        w = DepVar({'t':t, 'u':u}, {'v':v})
        self.assertAlmostEqual(w.value, 6.0)

        #test creation with scalar attributes
        w = DepVar(t=1.0, u=2.0, v=3.0)
        self.assertAlmostEqual(w.value, 6.0)

        #test invalid creation with no attributes
        with self.assertRaises(AssertionError):
            w = DepVar()

    def test_derivative(self):
        """Test derivative method."""
        t = relentless.variable.DesignVariable(value=1.0)
        u = relentless.variable.DesignVariable(value=2.0)
        v = relentless.variable.DesignVariable(value=3.0)

        #test derivative with 1 level of dependency
        w = DepVar(t=t, u=u, v=v)
        dw = w.derivative(t)
        self.assertAlmostEqual(dw, -0.5)
        dw = w.derivative(v)
        self.assertAlmostEqual(dw, -0.5)

        #test derivative with respect to self
        dw = w.derivative(w)
        self.assertAlmostEqual(dw, 1.0)

        #test derivative of variable not in the graph
        x = DepVar(t=t, v=v)
        dx = x.derivative(u)
        self.assertAlmostEqual(dx, 0.0)

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
    """Unit tests for relentless.variable.SameAs"""

    def test_init(self):
        """Test creation with data."""
        v = relentless.variable.DesignVariable(value=1.0)
        w = relentless.variable.SameAs(v)
        self.assertCountEqual(w.params, ('a',))

        u = relentless.variable.SameAs(w)
        z = relentless.variable.SameAs(2.0)

    def test_value(self):
        """Test compute method."""
        v = relentless.variable.DesignVariable(value=2.0)
        w = relentless.variable.SameAs(v)
        self.assertAlmostEqual(w.compute(a=1.0), 1.0)
        self.assertAlmostEqual(w.value, 2.0)

    def test_derivative(self):
        """Test compute_derivative method."""
        v = relentless.variable.DesignVariable(value=2.0)
        w = relentless.variable.SameAs(v)
        self.assertAlmostEqual(w.compute_derivative('a', a=1.0), 1.0)
        with self.assertRaises(ValueError):
            w.compute_derivative('x', a=1.0)
        self.assertAlmostEqual(w.derivative(v), 1.0)

class test_Negation(unittest.TestCase):
    """Unit tests for relentless.variable.Negation"""

    def test_init(self):
        """Test creation with data."""
        v = relentless.variable.DesignVariable(value=1.0)
        w = relentless.variable.Negation(v)
        self.assertCountEqual(w.params, ('a',))

        u = relentless.variable.Negation(w)
        z = relentless.variable.Negation(2.0)

    def test_value(self):
        """Test compute method."""
        v = relentless.variable.DesignVariable(value=2.0)
        w = relentless.variable.Negation(v)
        self.assertAlmostEqual(w.compute(a=1.0), -1.0)
        self.assertAlmostEqual(w.value, -2.0)

    def test_derivative(self):
        """Test compute_derivative method."""
        v = relentless.variable.DesignVariable(value=2.0)
        w = relentless.variable.Negation(v)
        self.assertAlmostEqual(w.compute_derivative('a', a=1.0), -1.0)
        with self.assertRaises(ValueError):
            w.compute_derivative('x', a=1.0)
        self.assertAlmostEqual(w.derivative(v), -1.0)

class test_Sum(unittest.TestCase):
    """Unit tests for relentless.variable.Sum"""

    def test_init(self):
        """Test creation with data."""
        u = relentless.variable.DesignVariable(value=1.0)
        v = relentless.variable.DesignVariable(value=2.0)
        w = relentless.variable.Sum(u, v)
        self.assertCountEqual(w.params, ('a','b'))

        x = relentless.variable.Sum(v, w)
        y = relentless.variable.Sum(w, x)
        w = relentless.variable.Sum(u, u)
        z = relentless.variable.Sum(2.0, 1.0)

    def test_value(self):
        """Test compute method."""
        u = relentless.variable.DesignVariable(value=1.0)
        v = relentless.variable.DesignVariable(value=2.0)
        w = relentless.variable.Sum(u, v)
        self.assertAlmostEqual(w.compute(a=4.0, b=3.0), 7.0)
        self.assertAlmostEqual(w.value, 3.0)

    def test_derivative(self):
        """Test compute_derivative method."""
        u = relentless.variable.DesignVariable(value=1.0)
        v = relentless.variable.DesignVariable(value=2.0)
        w = relentless.variable.Sum(u, v)
        self.assertAlmostEqual(w.compute_derivative('a', a=4.0, b=3.0), 1.0)
        self.assertAlmostEqual(w.compute_derivative('b', a=4.0, b=3.0), 1.0)
        with self.assertRaises(ValueError):
            w.compute_derivative('x', a=4.0, b=3.0)
        self.assertAlmostEqual(w.derivative(u), 1.0)
        self.assertAlmostEqual(w.derivative(v), 1.0)

class test_Difference(unittest.TestCase):
    """Unit tests for relentless.variable.Difference"""

    def test_init(self):
        """Test creation with data."""
        u = relentless.variable.DesignVariable(value=1.0)
        v = relentless.variable.DesignVariable(value=2.0)
        w = relentless.variable.Difference(u, v)
        self.assertCountEqual(w.params, ('a','b'))

        x = relentless.variable.Difference(v, w)
        y = relentless.variable.Difference(w, x)
        w = relentless.variable.Difference(u, u)
        z = relentless.variable.Difference(2.0, 1.0)

    def test_value(self):
        """Test compute method."""
        u = relentless.variable.DesignVariable(value=1.0)
        v = relentless.variable.DesignVariable(value=2.0)
        w = relentless.variable.Difference(u, v)
        self.assertAlmostEqual(w.compute(a=4.0, b=3.0), 1.0)
        self.assertAlmostEqual(w.value, -1.0)

    def test_derivative(self):
        """Test compute_derivative method."""
        u = relentless.variable.DesignVariable(value=1.0)
        v = relentless.variable.DesignVariable(value=2.0)
        w = relentless.variable.Difference(u, v)
        self.assertAlmostEqual(w.compute_derivative('a', a=4.0, b=3.0), 1.0)
        self.assertAlmostEqual(w.compute_derivative('b', a=4.0, b=3.0), -1.0)
        with self.assertRaises(ValueError):
            w.compute_derivative('x', a=4.0, b=3.0)
        self.assertAlmostEqual(w.derivative(u), 1.0)
        self.assertAlmostEqual(w.derivative(v), -1.0)

class test_Product(unittest.TestCase):
    """Unit tests for relentless.variable.Product"""

    def test_init(self):
        """Test creation with data."""
        u = relentless.variable.DesignVariable(value=1.0)
        v = relentless.variable.DesignVariable(value=2.0)
        w = relentless.variable.Product(u, v)
        self.assertCountEqual(w.params, ('a','b'))

        x = relentless.variable.Product(v, w)
        y = relentless.variable.Product(w, x)
        w = relentless.variable.Product(u, u)
        z = relentless.variable.Product(2.0, 1.0)

    def test_value(self):
        """Test compute method."""
        u = relentless.variable.DesignVariable(value=1.0)
        v = relentless.variable.DesignVariable(value=2.0)
        w = relentless.variable.Product(u, v)
        self.assertAlmostEqual(w.compute(a=4.0, b=3.0), 12.0)
        self.assertAlmostEqual(w.value, 2.0)

    def test_derivative(self):
        """Test compute_derivative method."""
        u = relentless.variable.DesignVariable(value=1.0)
        v = relentless.variable.DesignVariable(value=2.0)
        w = relentless.variable.Product(u, v)
        self.assertAlmostEqual(w.compute_derivative('a', a=4.0, b=3.0), 3.0)
        self.assertAlmostEqual(w.compute_derivative('b', a=4.0, b=3.0), 4.0)
        with self.assertRaises(ValueError):
            w.compute_derivative('x', a=4.0, b=3.0)
        self.assertAlmostEqual(w.derivative(u), 2.0)
        self.assertAlmostEqual(w.derivative(v), 1.0)

class test_Quotient(unittest.TestCase):
    """Unit tests for relentless.variable.Quotient"""

    def test_init(self):
        """Test creation with data."""
        u = relentless.variable.DesignVariable(value=1.0)
        v = relentless.variable.DesignVariable(value=2.0)
        w = relentless.variable.Quotient(u, v)
        self.assertCountEqual(w.params, ('a','b'))

        x = relentless.variable.Quotient(v, w)
        y = relentless.variable.Quotient(w, x)
        w = relentless.variable.Quotient(u, u)
        z = relentless.variable.Quotient(2.0, 1.0)

    def test_value(self):
        """Test compute method."""
        u = relentless.variable.DesignVariable(value=1.0)
        v = relentless.variable.DesignVariable(value=2.0)
        w = relentless.variable.Quotient(u, v)
        self.assertAlmostEqual(w.compute(a=9.0, b=3.0), 3.0)
        self.assertTrue(numpy.isnan(w.compute(a=1.0, b=0.0)))
        self.assertAlmostEqual(w.value, 0.5)

    def test_derivative(self):
        """Test compute_derivative method."""
        u = relentless.variable.DesignVariable(value=1.0)
        v = relentless.variable.DesignVariable(value=2.0)
        w = relentless.variable.Quotient(u, v)
        self.assertAlmostEqual(w.compute_derivative('a', a=9.0, b=3.0), 1.0/3.0)
        self.assertTrue(numpy.isnan(w.compute_derivative('a', a=1.0, b=0.0)))
        self.assertAlmostEqual(w.compute_derivative('b', a=9.0, b=3.0), -1.0)
        self.assertTrue(numpy.isnan(w.compute_derivative('b', a=1.0, b=0.0)))
        with self.assertRaises(ValueError):
            w.compute_derivative('x', a=4.0, b=3.0)
        self.assertAlmostEqual(w.derivative(u), 1.0/2.0)
        self.assertAlmostEqual(w.derivative(v), -1.0/4.0)

class test_Power(unittest.TestCase):
    """Unit tests for relentless.variable.Power"""

    def test_init(self):
        """Test creation with data."""
        u = relentless.variable.DesignVariable(value=1.0)
        v = relentless.variable.DesignVariable(value=2.0)
        w = relentless.variable.Power(u, v)
        self.assertCountEqual(w.params, ('a','b'))

        x = relentless.variable.Power(v, w)
        y = relentless.variable.Power(w, x)
        w = relentless.variable.Power(u, u)
        z = relentless.variable.Power(2.0, 1.0)

    def test_value(self):
        """Test compute method."""
        u = relentless.variable.DesignVariable(value=2.0)
        v = relentless.variable.DesignVariable(value=3)
        w = relentless.variable.Power(u, v)
        self.assertAlmostEqual(w.compute(a=4.0, b=0.5), 2.0)
        self.assertAlmostEqual(w.compute(a=4.0, b=0), 1.0)
        self.assertAlmostEqual(w.compute(a=4.0, b=-0.5), 0.5)
        self.assertAlmostEqual(w.compute(a=0.0, b=2), 0.0)
        self.assertAlmostEqual(w.value, 8.0)

    def test_derivative(self):
        """Test compute_derivative method."""
        u = relentless.variable.DesignVariable(value=2.0)
        v = relentless.variable.DesignVariable(value=3.0)
        w = relentless.variable.Power(u, v)
        self.assertAlmostEqual(w.compute_derivative('a', a=4.0, b=0.5), 1./4.)
        self.assertAlmostEqual(w.compute_derivative('a', a=4.0, b=0.0), 0.0)
        self.assertAlmostEqual(w.compute_derivative('a', a=4.0, b=-0.5), -1./16)
        self.assertAlmostEqual(w.compute_derivative('a', a=0.0, b=2), 0.0)
        self.assertAlmostEqual(w.compute_derivative('b', a=4.0, b=0.5), 2.0*numpy.log(4.0))
        self.assertAlmostEqual(w.compute_derivative('b', a=4.0, b=0.0), numpy.log(4.0))
        self.assertAlmostEqual(w.compute_derivative('b', a=4.0, b=-0.5), 0.5*numpy.log(4.0))
        self.assertAlmostEqual(w.compute_derivative('a', a=0.0, b=2), 0.0)
        with self.assertRaises(ValueError):
            w.compute_derivative('x', a=4.0, b=3.0)
        self.assertAlmostEqual(w.derivative(u), 12)
        self.assertAlmostEqual(w.derivative(v), 8.0*numpy.log(2.0))

class test_ArithmeticMean(unittest.TestCase):
    """Unit tests for relentless.variable.ArithmeticMean"""

    def test_init(self):
        """Test creation with data."""
        u = relentless.variable.DesignVariable(value=1.0)
        v = relentless.variable.DesignVariable(value=2.0)
        w = relentless.variable.ArithmeticMean(u, v)
        self.assertCountEqual(w.params, ('a','b'))

        x = relentless.variable.ArithmeticMean(v, w)
        y = relentless.variable.ArithmeticMean(w, x)
        w = relentless.variable.ArithmeticMean(u, u)
        z = relentless.variable.ArithmeticMean(2.0, 1.0)

    def test_value(self):
        """Test compute method."""
        u = relentless.variable.DesignVariable(value=1.0)
        v = relentless.variable.DesignVariable(value=2.0)
        w = relentless.variable.ArithmeticMean(u, v)
        self.assertAlmostEqual(w.compute(a=4.0, b=3.0), 3.5)
        self.assertAlmostEqual(w.value, 1.5)

    def test_derivative(self):
        """Test compute_derivative method."""
        u = relentless.variable.DesignVariable(value=1.0)
        v = relentless.variable.DesignVariable(value=2.0)
        w = relentless.variable.ArithmeticMean(u, v)
        self.assertAlmostEqual(w.compute_derivative('a', a=4.0, b=3.0), 0.5)
        self.assertAlmostEqual(w.compute_derivative('b', a=4.0, b=3.0), 0.5)
        with self.assertRaises(ValueError):
            w.compute_derivative('x', a=4.0, b=3.0)
        self.assertAlmostEqual(w.derivative(u), 0.5)
        self.assertAlmostEqual(w.derivative(v), 0.5)

class test_GeometricMean(unittest.TestCase):
    """Unit tests for relentless.variable.GeometricMean"""

    def test_init(self):
        """Test creation with data."""
        u = relentless.variable.DesignVariable(value=1.0)
        v = relentless.variable.DesignVariable(value=16.0)
        w = relentless.variable.GeometricMean(u, v)
        self.assertCountEqual(w.params, ('a','b'))

        x = relentless.variable.GeometricMean(v, w)
        y = relentless.variable.GeometricMean(w, x)
        w = relentless.variable.GeometricMean(u, u)
        z = relentless.variable.GeometricMean(2.0, 1.0)

    def test_value(self):
        """Test compute method."""
        u = relentless.variable.DesignVariable(value=1.0)
        v = relentless.variable.DesignVariable(value=16.0)
        w = relentless.variable.GeometricMean(u, v)
        self.assertAlmostEqual(w.compute(a=9.0, b=4.0), 6.0)
        self.assertAlmostEqual(w.value, 4.0)

    def test_derivative(self):
        """Test compute_derivative method."""
        u = relentless.variable.DesignVariable(value=1.0)
        v = relentless.variable.DesignVariable(value=16.0)
        w = relentless.variable.GeometricMean(u, v)
        self.assertAlmostEqual(w.compute_derivative('a', a=9.0, b=4.0), 0.5*2.0/3.0)
        self.assertTrue(numpy.isnan(w.compute_derivative('a', a=0.0, b=4.0)))
        self.assertAlmostEqual(w.compute_derivative('a', a=9.0, b=0.0),0.0)
        self.assertAlmostEqual(w.compute_derivative('b', a=9.0, b=4.0), 0.5*3.0/2.0)
        self.assertTrue(numpy.isnan(w.compute_derivative('b', a=9.0, b=0.0)))
        self.assertAlmostEqual(w.compute_derivative('b', a=0.0, b=4.0),0.0)
        with self.assertRaises(ValueError):
            w.compute_derivative('x', a=9.0, b=4.0)
        self.assertAlmostEqual(w.derivative(u), 0.5*4.0/1.0)
        self.assertAlmostEqual(w.derivative(v), 0.5*1.0/4.0)

if __name__ == '__main__':
    unittest.main()
