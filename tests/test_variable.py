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
        self.assertEqual(str(x), "4.0")

class test_IndependentVariable(unittest.TestCase):
    """Unit tests for relentless.variable.IndependentVariable."""

    def test_init(self):
        """Test construction with data."""
        #test scalar values
        v = relentless.variable.IndependentVariable(1.0)
        self.assertEqual(v.value, 1.0)

        v = relentless.variable.IndependentVariable('1.0')
        self.assertEqual(v.value, '1.0')

        v = relentless.variable.IndependentVariable(True)
        self.assertEqual(v.value, True)

        v = relentless.variable.IndependentVariable(None)
        self.assertEqual(v.value, None)

        #test "array" values
        v = relentless.variable.IndependentVariable([1,2])
        self.assertEqual(v.value, [1,2])

        v = relentless.variable.IndependentVariable((1,2))
        self.assertEqual(v.value, (1,2))

        v = relentless.variable.IndependentVariable({'1':1,'2':2})
        self.assertEqual(v.value, {'1':1,'2':2})

        v = relentless.variable.IndependentVariable(set([1,2]))
        self.assertEqual(v.value, set([1,2]))

    def test_value(self):
        """Test value setting."""
        #test scalar values
        v = relentless.variable.IndependentVariable(1.0)
        self.assertEqual(v.value, 1.0)

        v.value = '1.0'
        self.assertEqual(v.value, '1.0')

        v.value = True
        self.assertEqual(v.value, True)

        v.value = None
        self.assertEqual(v.value, None)

        #test "array" values
        v.value = [1,2]
        self.assertEqual(v.value, [1,2])

        v.value = (1,2)
        self.assertEqual(v.value, (1,2))

        v.value = {'1':1, '2':2}
        self.assertEqual(v.value, {'1':1,'2':2})

        v.value = set([1,2])
        self.assertEqual(v.value, set([1,2]))

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
        #test with no bounds and non-default value of `const`
        v = relentless.variable.DesignVariable(value=1.0, const=True)
        self.assertAlmostEqual(v.value, 1.0)
        self.assertEqual(v.const, True)
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
        with self.assertRaises(ValueError):
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

    @property
    def value(self):
        return numpy.sum([v.value for p,v in self.depends])

    def _derivative(self, param):
        #Note: this method doesn't calculate the actual derivatives;
        #it is merely used for testing the chain rule calculations
        #with various variable dependencies
        if param in self.params:
            if getattr(self, param).value > 5.0:
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
        self.assertCountEqual(w.params, ('t','u','v'))
        self.assertDictEqual({p:v for p,v in w.depends}, {'t':t,'u':u,'v':v})
        self.assertAlmostEqual(w.value, 6.0)

        #test creation with only kwvars
        w = DepVar(t=t, u=u, v=v)
        self.assertCountEqual(w.params, ('t','u','v'))
        self.assertDictEqual({p:v for p,v in w.depends}, {'t':t,'u':u,'v':v})
        self.assertAlmostEqual(w.value, 6.0)

        #test creation with vardicts and kwvars
        w = DepVar({'t':t, 'u':u}, v=v)
        self.assertCountEqual(w.params, ('t','u','v'))
        self.assertDictEqual({p:v for p,v in w.depends}, {'t':t,'u':u,'v':v})
        self.assertAlmostEqual(w.value, 6.0)

        #test creation with repeated attributes
        w = DepVar({'t':t, 'u':u}, u=v)
        self.assertCountEqual(w.params, ('t','u'))
        self.assertDictEqual({p:v for p,v in w.depends}, {'t':t,'u':v})
        self.assertAlmostEqual(w.value, 4.0)

        #test creation with multiple vardicts
        w = DepVar({'t':t, 'u':u}, {'v':v})
        self.assertCountEqual(w.params, ('t','u','v'))
        self.assertDictEqual({p:v for p,v in w.depends}, {'t':t,'u':u,'v':v})
        self.assertAlmostEqual(w.value, 6.0)

        #test creation with scalar attributes
        w = DepVar(t=1.0, u=2.0, v=3.0)
        self.assertCountEqual(w.params, ('t','u','v'))
        self.assertDictEqual({p:v.value for p,v in w.depends}, {'t':1.0,'u':2.0,'v':3.0})
        self.assertAlmostEqual(w.value, 6.0)

        #change scalar attribute value
        w.t = 4.0
        self.assertCountEqual(w.params, ('t','u','v'))
        self.assertDictEqual({p:v.value for p,v in w.depends}, {'t':4.0,'u':2.0,'v':3.0})
        self.assertAlmostEqual(w.value, 9.0)

        #test invalid creation with no attributes
        with self.assertRaises(AttributeError):
            w = DepVar()

    def test_dependency_graph(self):
        """Test dependency_graph method."""
        t = relentless.variable.DesignVariable(value=1.0)
        u = relentless.variable.DesignVariable(value=2.0)
        v = relentless.variable.DesignVariable(value=3.0)

        #test 1 level of dependency
        w = DepVar(t=t, u=u, v=v)
        g = w.dependency_graph()
        self.assertCountEqual(g.nodes, [t,u,v,w])
        self.assertCountEqual(g.edges, [(w,t),(w,u),(w,v)])
        self.assertEqual(g.edges[w,t]['params'], ['t'])
        self.assertEqual(g.edges[w,u]['params'], ['u'])
        self.assertEqual(g.edges[w,v]['params'], ['v'])

        #test 2 levels of dependency
        x = DepVar(t=t, v=v)
        y = DepVar(w=w, x=x)
        g = y.dependency_graph()
        self.assertCountEqual(g.nodes, [t,u,v,w,x,y])
        self.assertCountEqual(g.edges, [(w,t),(w,u),(w,v),(x,t),(x,v),(y,w),(y,x)])
        self.assertEqual(g.edges[w,t]['params'], ['t'])
        self.assertEqual(g.edges[w,u]['params'], ['u'])
        self.assertEqual(g.edges[w,v]['params'], ['v'])
        self.assertEqual(g.edges[x,t]['params'], ['t'])
        self.assertEqual(g.edges[x,v]['params'], ['v'])
        self.assertEqual(g.edges[y,w]['params'], ['w'])
        self.assertEqual(g.edges[y,x]['params'], ['x'])

        #test more complex dependency
        z = DepVar(t=t, w=w, y=y)
        g = z.dependency_graph()
        self.assertCountEqual(g.nodes, [t,u,v,w,x,y,z])
        self.assertCountEqual(g.edges, [(w,t),(w,u),(w,v),(x,t),(x,v),(y,w),(y,x),
                                        (z,t),(z,w),(z,y)])

        #test that dependencies are acyclic
        g_acyclic = relentless.variable.DependentVariable._assert_acyclic(g)
        self.assertEqual(g_acyclic, g)

        #test 'multiple' dependency on same object
        q = DepVar(t=t, u=t, v=t)
        g = q.dependency_graph()
        self.assertCountEqual(g.nodes, [t,q])
        self.assertCountEqual(g.edges, [(q,t)])
        self.assertCountEqual(g.edges[q,t]['params'], ['t','u','v'])

        q = DepVar(t=t, u=t, v=v, w=v)
        g = q.dependency_graph()
        self.assertCountEqual(g.nodes, [t,v,q])
        self.assertCountEqual(g.edges, [(q,t),(q,v)])
        self.assertCountEqual(g.edges[q,t]['params'], ['t','u'])
        self.assertCountEqual(g.edges[q,v]['params'], ['v','w'])

        #test circular dependencies
        a = DepVar(t=t)
        b = DepVar(t=a)
        a.t = b
        g = a.dependency_graph()
        self.assertCountEqual(g.nodes, [a,b])
        self.assertCountEqual(g.edges, [(a,b),(b,a)])
        self.assertCountEqual(g.edges[a,b]['params'], ['t'])
        self.assertCountEqual(g.edges[b,a]['params'], ['t'])

        a = DepVar(t=t)
        b = DepVar(t=a)
        c = DepVar(t=b)
        a.t = c
        g = a.dependency_graph()
        self.assertCountEqual(g.nodes, [a,b,c])
        self.assertCountEqual(g.edges, [(b,a),(c,b),(a,c)])

        a = DepVar(t=t)
        c = DepVar(t=t)
        b = DepVar(t=a, u=c)
        a.t = b
        g = a.dependency_graph()
        self.assertCountEqual(g.nodes, [t,a,b,c])
        self.assertCountEqual(g.edges, [(b,a),(a,b),(b,c),(c,t)])

        #test that dependencies are not acyclic
        with self.assertRaises(RuntimeError):
            relentless.variable.DependentVariable._assert_acyclic(g)

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

        #test derivative with circular dependencies
        a = DepVar(t=t)
        a.t = DepVar(t=a)
        with self.assertRaises(RuntimeError):
            a.derivative(t)

        a = DepVar(t=t)
        b = DepVar(t=a)
        c = DepVar(t=b)
        a.t = DepVar(t=c)
        with self.assertRaises(RuntimeError):
            a.derivative(t)

class test_SameAs(unittest.TestCase):
    """Unit tests for relentless.variable.SameAs"""

    def test_init(self):
        """Test creation with data."""
        #test variable dependent on DesignVariable
        v = relentless.variable.DesignVariable(value=1.0)
        w = relentless.variable.SameAs(v)
        self.assertEqual(w.value, 1.0)
        self.assertCountEqual(w.params, ('a',))
        self.assertDictEqual({p:v for p,v in w.depends}, {'a':v})

        #test variable dependent on DependentVariable
        u = relentless.variable.SameAs(w)
        self.assertEqual(u.value, 1.0)
        self.assertCountEqual(u.params, ('a',))
        self.assertDictEqual({p:v for p,v in u.depends}, {'a':w})

        #test scalar dependency
        z = relentless.variable.SameAs(2.0)
        self.assertAlmostEqual(z.value, 2.0)
        self.assertCountEqual(z.params, ('a',))
        self.assertDictEqual({p:v.value for p,v in z.depends}, {'a':2.0})

    def test_value(self):
        """Test setting values."""
        #create "chained" dependent variables
        v = relentless.variable.DesignVariable(value=1.0, low=0.5, high=1.5)
        w = relentless.variable.SameAs(v)
        x = relentless.variable.SameAs(w)

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
        v = relentless.variable.DesignVariable(value=2.0)
        w = relentless.variable.SameAs(v)

        #test w.r.t. a
        dw = w._derivative('a')
        self.assertEqual(dw, 1.0)

        #test w.r.t. ~a
        with self.assertRaises(ValueError):
            w._derivative('x')

        #change value and retest derivative
        v.value = 3.0
        dw = w._derivative('a')
        self.assertEqual(dw, 1.0)

class test_Negation(unittest.TestCase):
    """Unit tests for relentless.variable.Negation"""

    def test_init(self):
        """Test creation with data."""
        #test variable dependent on DesignVariable
        v = relentless.variable.DesignVariable(value=1.0)
        w = relentless.variable.Negation(v)
        self.assertEqual(w.value, -1.0)
        self.assertCountEqual(w.params, ('a',))
        self.assertDictEqual({p:v for p,v in w.depends}, {'a':v})

        #test variable dependent on DependentVariable
        u = relentless.variable.Negation(w)
        self.assertEqual(u.value, 1.0)
        self.assertCountEqual(u.params, ('a',))
        self.assertDictEqual({p:v for p,v in u.depends}, {'a':w})

        #test scalar dependency
        z = relentless.variable.Negation(2.0)
        self.assertAlmostEqual(z.value, -2.0)
        self.assertCountEqual(z.params, ('a',))
        self.assertDictEqual({p:v.value for p,v in z.depends}, {'a':2.0})

    def test_value(self):
        """Test setting values."""
        #create "chained" dependent variables
        v = relentless.variable.DesignVariable(value=1.0, low=0.5, high=1.5)
        w = relentless.variable.Negation(v)
        x = relentless.variable.Negation(w)

        self.assertEqual(v.value, 1.0)
        self.assertEqual(w.value, -1.0)
        self.assertEqual(x.value, 1.0)

        #change value of v
        v.value = 0.4
        self.assertEqual(v.value, 0.5)
        self.assertEqual(w.value, -0.5)
        self.assertEqual(x.value, 0.5)

    def test_derivative(self):
        """Test _derivative method."""
        v = relentless.variable.DesignVariable(value=2.0)
        w = relentless.variable.Negation(v)

        #test w.r.t. a
        dw = w._derivative('a')
        self.assertEqual(dw, -1.0)

        #test w.r.t. ~a
        with self.assertRaises(ValueError):
            w._derivative('x')

        #change value and retest derivative
        v.value = 3.0
        dw = w._derivative('a')
        self.assertEqual(dw, -1.0)

class test_Sum(unittest.TestCase):
    """Unit tests for relentless.variable.Sum"""

    def test_init(self):
        """Test creation with data."""
        #test variable dependent on 2 DesignVariables
        u = relentless.variable.DesignVariable(value=1.0)
        v = relentless.variable.DesignVariable(value=2.0)
        w = relentless.variable.Sum(u, v)
        self.assertAlmostEqual(w.value, 3.0)
        self.assertCountEqual(w.params, ('a','b'))
        self.assertDictEqual({p:v for p,v in w.depends}, {'a':u,'b':v})

        #test variable dependent on 1 DesignVar, 1 DependentVar
        x = relentless.variable.Sum(v,w)
        self.assertAlmostEqual(x.value, 5.0)
        self.assertCountEqual(x.params, ('a','b'))
        self.assertDictEqual({p:v for p,v in x.depends}, {'a':v,'b':w})

        #test variable dependent on 2 DependentVariables
        y = relentless.variable.Sum(w,x)
        self.assertAlmostEqual(y.value, 8.0)
        self.assertCountEqual(y.params, ('a','b'))
        self.assertDictEqual({p:v for p,v in y.depends}, {'a':w,'b':x})

        #test same-object dependencies
        w = relentless.variable.Sum(u, u)
        self.assertAlmostEqual(w.value, 2.0)
        self.assertCountEqual(w.params, ('a','b'))
        self.assertDictEqual({p:v for p,v in w.depends}, {'a':u,'b':u})

        #test scalar dependencies
        z = relentless.variable.Sum(2.0, 1.0)
        self.assertAlmostEqual(z.value, 3.0)
        self.assertCountEqual(z.params, ('a','b'))
        self.assertDictEqual({p:v.value for p,v in z.depends}, {'a':2.0,'b':1.0})

    def test_value(self):
        """Test setting values."""
        #create "chained" dependent variables
        u = relentless.variable.DesignVariable(value=2.0, low=1.5, high=2.5)
        v = relentless.variable.DesignVariable(value=1.0, low=0.5, high=1.5)
        w = relentless.variable.Sum(u, v)
        x = relentless.variable.Sum(v, w)
        y = relentless.variable.Sum(w, x)

        self.assertEqual(u.value, 2.0)
        self.assertEqual(v.value, 1.0)
        self.assertAlmostEqual(w.value, 3.0)
        self.assertAlmostEqual(x.value, 4.0)
        self.assertAlmostEqual(y.value, 7.0)

        #change value of u
        u.value = 2.6
        self.assertEqual(u.value, 2.5)
        self.assertEqual(v.value, 1.0)
        self.assertAlmostEqual(w.value, 3.5)
        self.assertAlmostEqual(x.value, 4.5)
        self.assertAlmostEqual(y.value, 8.0)

        #change value of v
        v.value = 0.4
        self.assertEqual(u.value, 2.5)
        self.assertEqual(v.value, 0.5)
        self.assertAlmostEqual(w.value, 3.0)
        self.assertAlmostEqual(x.value, 3.5)
        self.assertAlmostEqual(y.value, 6.5)

    def test_derivative(self):
        """Test _derivative method."""
        u = relentless.variable.DesignVariable(value=1.0)
        v = relentless.variable.DesignVariable(value=2.0)
        w = relentless.variable.Sum(u, v)

        #test w.r.t. a
        dw = w._derivative('a')
        self.assertEqual(dw, 1.0)

        #test w.r.t. b
        dw = w._derivative('b')
        self.assertEqual(dw, 1.0)

        #test w.r.t. ~a,~b
        with self.assertRaises(ValueError):
            w._derivative('c')

        #change values and retest derivative
        u.value = 1.5
        v.value = 2.5
        dw = w._derivative('b')
        self.assertEqual(dw, 1.0)

class test_Difference(unittest.TestCase):
    """Unit tests for relentless.variable.Difference"""

    def test_init(self):
        """Test creation with data."""
        #test variable dependent on 2 DesignVariables
        u = relentless.variable.DesignVariable(value=1.0)
        v = relentless.variable.DesignVariable(value=2.0)
        w = relentless.variable.Difference(u, v)
        self.assertAlmostEqual(w.value, -1.0)
        self.assertCountEqual(w.params, ('a','b'))
        self.assertDictEqual({p:v for p,v in w.depends}, {'a':u,'b':v})

        #test variable dependent on 1 DesignVar, 1 DependentVar
        x = relentless.variable.Difference(v,w)
        self.assertAlmostEqual(x.value, 3.0)
        self.assertCountEqual(x.params, ('a','b'))
        self.assertDictEqual({p:v for p,v in x.depends}, {'a':v,'b':w})

        #test variable dependent on 2 DependentVariables
        y = relentless.variable.Difference(w,x)
        self.assertAlmostEqual(y.value, -4.0)
        self.assertCountEqual(y.params, ('a','b'))
        self.assertDictEqual({p:v for p,v in y.depends}, {'a':w,'b':x})

        #test same-object dependencies
        w = relentless.variable.Difference(u, u)
        self.assertAlmostEqual(w.value, 0.0)
        self.assertCountEqual(w.params, ('a','b'))
        self.assertDictEqual({p:v for p,v in w.depends}, {'a':u,'b':u})

        #test scalar dependencies
        z = relentless.variable.Difference(2.0, 1.0)
        self.assertAlmostEqual(z.value, 1.0)
        self.assertCountEqual(z.params, ('a','b'))
        self.assertDictEqual({p:v.value for p,v in z.depends}, {'a':2.0,'b':1.0})

    def test_value(self):
        """Test setting values."""
        #create "chained" dependent variables
        u = relentless.variable.DesignVariable(value=2.0, low=1.5, high=2.5)
        v = relentless.variable.DesignVariable(value=1.0, low=0.5, high=1.5)
        w = relentless.variable.Difference(u, v)
        x = relentless.variable.Difference(v, w)
        y = relentless.variable.Difference(w, x)

        self.assertEqual(u.value, 2.0)
        self.assertEqual(v.value, 1.0)
        self.assertAlmostEqual(w.value, 1.0)
        self.assertAlmostEqual(x.value, 0.0)
        self.assertAlmostEqual(y.value, 1.0)

        #change value of u
        u.value = 2.6
        self.assertEqual(u.value, 2.5)
        self.assertEqual(v.value, 1.0)
        self.assertAlmostEqual(w.value, 1.5)
        self.assertAlmostEqual(x.value, -0.5)
        self.assertAlmostEqual(y.value, 2.0)

        #change value of v
        v.value = 0.4
        self.assertEqual(u.value, 2.5)
        self.assertEqual(v.value, 0.5)
        self.assertAlmostEqual(w.value, 2.0)
        self.assertAlmostEqual(x.value, -1.5)
        self.assertAlmostEqual(y.value, 3.5)

    def test_derivative(self):
        """Test _derivative method."""
        u = relentless.variable.DesignVariable(value=1.0)
        v = relentless.variable.DesignVariable(value=2.0)
        w = relentless.variable.Difference(u, v)

        #test w.r.t. a
        dw = w._derivative('a')
        self.assertEqual(dw, 1.0)

        #test w.r.t. b
        dw = w._derivative('b')
        self.assertEqual(dw, -1.0)

        #test w.r.t. ~a,~b
        with self.assertRaises(ValueError):
            w._derivative('c')

        #change values and retest derivative
        u.value = 1.5
        v.value = 2.5
        dw = w._derivative('b')
        self.assertEqual(dw, -1.0)

class test_Product(unittest.TestCase):
    """Unit tests for relentless.variable.Product"""

    def test_init(self):
        """Test creation with data."""
        #test variable dependent on 2 DesignVariables
        u = relentless.variable.DesignVariable(value=2.0)
        v = relentless.variable.DesignVariable(value=3.0)
        w = relentless.variable.Product(u, v)
        self.assertAlmostEqual(w.value, 6.0)
        self.assertCountEqual(w.params, ('a','b'))
        self.assertDictEqual({p:v for p,v in w.depends}, {'a':u,'b':v})

        #test variable dependent on 1 DesignVar, 1 DependentVar
        x = relentless.variable.Product(v,w)
        self.assertAlmostEqual(x.value, 18.0)
        self.assertCountEqual(x.params, ('a','b'))
        self.assertDictEqual({p:v for p,v in x.depends}, {'a':v,'b':w})

        #test variable dependent on 2 DependentVariables
        y = relentless.variable.Product(w,x)
        self.assertAlmostEqual(y.value, 108.0)
        self.assertCountEqual(y.params, ('a','b'))
        self.assertDictEqual({p:v for p,v in y.depends}, {'a':w,'b':x})

        #test same-object dependencies
        w = relentless.variable.Product(u, u)
        self.assertAlmostEqual(w.value, 4.0)
        self.assertCountEqual(w.params, ('a','b'))
        self.assertDictEqual({p:v for p,v in w.depends}, {'a':u,'b':u})

        #test scalar dependencies
        z = relentless.variable.Product(2.0, 1.5)
        self.assertAlmostEqual(z.value, 3.0)
        self.assertCountEqual(z.params, ('a','b'))
        self.assertDictEqual({p:v.value for p,v in z.depends}, {'a':2.0,'b':1.5})

    def test_value(self):
        """Test setting values."""
        #create "chained" dependent variables
        u = relentless.variable.DesignVariable(value=2.0, low=1.5, high=2.5)
        v = relentless.variable.DesignVariable(value=1.5, low=0.5, high=1.5)
        w = relentless.variable.Product(u, v)
        x = relentless.variable.Product(v, w)
        y = relentless.variable.Product(w, x)

        self.assertEqual(u.value, 2.0)
        self.assertEqual(v.value, 1.5)
        self.assertAlmostEqual(w.value, 3.0)
        self.assertAlmostEqual(x.value, 4.5)
        self.assertAlmostEqual(y.value, 13.5)

        #change value of u
        u.value = 2.6
        self.assertEqual(u.value, 2.5)
        self.assertEqual(v.value, 1.5)
        self.assertAlmostEqual(w.value, 3.75)
        self.assertAlmostEqual(x.value, 5.625)
        self.assertAlmostEqual(y.value, 21.09375)

        #change value of v
        v.value = 0.4
        self.assertEqual(u.value, 2.5)
        self.assertEqual(v.value, 0.5)
        self.assertAlmostEqual(w.value, 1.25)
        self.assertAlmostEqual(x.value, 0.625)
        self.assertAlmostEqual(y.value, 0.78125)

    def test_derivative(self):
        """Test _derivative method."""
        u = relentless.variable.DesignVariable(value=1.5)
        v = relentless.variable.DesignVariable(value=2.0)
        w = relentless.variable.Product(u,v)

        #test w.r.t. a
        dw = w._derivative('a')
        self.assertAlmostEqual(dw, 2.0)

        #test w.r.t. b
        dw = w._derivative('b')
        self.assertAlmostEqual(dw, 1.5)

        #test w.r.t. ~a,~b
        with self.assertRaises(ValueError):
            w._derivative('c')

        #change values and retest derivative
        u.value = 1.5
        v.value = 2.5
        dw = w._derivative('a')
        self.assertAlmostEqual(dw, 2.5)

class test_Quotient(unittest.TestCase):
    """Unit tests for relentless.variable.Quotient"""

    def test_init(self):
        """Test creation with data."""
        #test variable dependent on 2 DesignVariables
        u = relentless.variable.DesignVariable(value=6.0)
        v = relentless.variable.DesignVariable(value=3.0)
        w = relentless.variable.Quotient(u, v)
        self.assertAlmostEqual(w.value, 2.0)
        self.assertCountEqual(w.params, ('a','b'))
        self.assertDictEqual({p:v for p,v in w.depends}, {'a':u,'b':v})

        #test variable dependent on 1 DesignVar, 1 DependentVar
        x = relentless.variable.Quotient(v,w)
        self.assertAlmostEqual(x.value, 1.5)
        self.assertCountEqual(x.params, ('a','b'))
        self.assertDictEqual({p:v for p,v in x.depends}, {'a':v,'b':w})

        #test variable dependent on 2 DependentVariables
        y = relentless.variable.Quotient(w,x)
        self.assertAlmostEqual(y.value, 1.333333333)
        self.assertCountEqual(y.params, ('a','b'))
        self.assertDictEqual({p:v for p,v in y.depends}, {'a':w,'b':x})

        #test same-object dependencies
        w = relentless.variable.Quotient(u, u)
        self.assertAlmostEqual(w.value, 1.0)
        self.assertCountEqual(w.params, ('a','b'))
        self.assertDictEqual({p:v for p,v in w.depends}, {'a':u,'b':u})

        #test scalar dependencies
        z = relentless.variable.Quotient(3.0, 1.5)
        self.assertAlmostEqual(z.value, 2.0)
        self.assertCountEqual(z.params, ('a','b'))
        self.assertDictEqual({p:v.value for p,v in z.depends}, {'a':3.0,'b':1.5})

        #test with 0 denominator
        a = relentless.variable.Quotient(1.0, 0.0)
        self.assertEqual(a.value, numpy.inf)

    def test_value(self):
        """Test setting values."""
        #create "chained" dependent variables
        u = relentless.variable.DesignVariable(value=2.0, low=1.5, high=2.5)
        v = relentless.variable.DesignVariable(value=1.25, low=0.5, high=1.5)
        w = relentless.variable.Quotient(u, v)
        x = relentless.variable.Quotient(v, w)
        y = relentless.variable.Quotient(w, x)

        self.assertEqual(u.value, 2.0)
        self.assertEqual(v.value, 1.25)
        self.assertAlmostEqual(w.value, 1.6)
        self.assertAlmostEqual(x.value, 0.78125)
        self.assertAlmostEqual(y.value, 2.048)

        #change value of u
        u.value = 2.6
        self.assertEqual(u.value, 2.5)
        self.assertEqual(v.value, 1.25)
        self.assertAlmostEqual(w.value, 2.0)
        self.assertAlmostEqual(x.value, 0.625)
        self.assertAlmostEqual(y.value, 3.2)

        #change value of v
        v.value = 0.4
        self.assertEqual(u.value, 2.5)
        self.assertEqual(v.value, 0.5)
        self.assertAlmostEqual(w.value, 5.0)
        self.assertAlmostEqual(x.value, 0.1)
        self.assertAlmostEqual(y.value, 50.0)

    def test_derivative(self):
        """Test _derivative method."""
        u = relentless.variable.DesignVariable(value=1.5)
        v = relentless.variable.DesignVariable(value=2.0)
        w = relentless.variable.Quotient(u,v)

        #test w.r.t. a
        dw = w._derivative('a')
        self.assertAlmostEqual(dw, 0.5)

        #test w.r.t. b
        dw = w._derivative('b')
        self.assertAlmostEqual(dw, -0.375)

        #test w.r.t. ~a,~b
        with self.assertRaises(ValueError):
            w._derivative('c')

        #test with 0 denominator
        v.value = 0.0
        dw = w._derivative('a')
        self.assertEqual(dw, numpy.inf)
        dw = w._derivative('b')
        self.assertEqual(dw, numpy.inf)

class test_Power(unittest.TestCase):
    """Unit tests for relentless.variable.Power"""

    def test_init(self):
        """Test creation with data."""
        #test variable dependent on 2 DesignVariables
        u = relentless.variable.DesignVariable(value=2.0)
        v = relentless.variable.DesignVariable(value=2.0)
        w = relentless.variable.Power(u, v)
        self.assertAlmostEqual(w.value, 4.0)
        self.assertCountEqual(w.params, ('a','b'))
        self.assertDictEqual({p:v for p,v in w.depends}, {'a':u,'b':v})

        #test variable dependent on 1 DesignVar, 1 DependentVar
        x = relentless.variable.Power(v,w)
        self.assertAlmostEqual(x.value, 16.0)
        self.assertCountEqual(x.params, ('a','b'))
        self.assertDictEqual({p:v for p,v in x.depends}, {'a':v,'b':w})

        #test variable dependent on 2 DependentVariables
        y = relentless.variable.Power(w,x)
        self.assertAlmostEqual(y.value, 4294967296.0)
        self.assertCountEqual(y.params, ('a','b'))
        self.assertDictEqual({p:v for p,v in y.depends}, {'a':w,'b':x})

        #test same-object dependencies
        w = relentless.variable.Power(u, u)
        self.assertAlmostEqual(w.value, 4.0)
        self.assertCountEqual(w.params, ('a','b'))
        self.assertDictEqual({p:v for p,v in w.depends}, {'a':u,'b':u})

        #test scalar dependencies
        z = relentless.variable.Power(2.0, 3.0)
        self.assertAlmostEqual(z.value, 8.0)
        self.assertCountEqual(z.params, ('a','b'))
        self.assertDictEqual({p:v.value for p,v in z.depends}, {'a':2.0,'b':3.0})

    def test_value(self):
        """Test setting values."""
        #create "chained" dependent variables
        u = relentless.variable.DesignVariable(value=2.0, low=1.5, high=2.5)
        v = relentless.variable.DesignVariable(value=1.5, low=0.5, high=1.5)
        w = relentless.variable.Power(u, v)
        x = relentless.variable.Power(v, w)
        y = relentless.variable.Power(w, x)

        self.assertEqual(u.value, 2.0)
        self.assertEqual(v.value, 1.5)
        self.assertAlmostEqual(w.value, 2.828427125)
        self.assertAlmostEqual(x.value, 3.148192599)
        self.assertAlmostEqual(y.value, 26.39675894)

        #change value of u
        u.value = 2.6
        self.assertEqual(u.value, 2.5)
        self.assertEqual(v.value, 1.5)
        self.assertAlmostEqual(w.value, 3.952847075)
        self.assertAlmostEqual(x.value, 4.966630124)
        self.assertAlmostEqual(y.value, 921.7884284)

        #change value of v
        v.value = 0.4
        self.assertEqual(u.value, 2.5)
        self.assertEqual(v.value, 0.5)
        self.assertAlmostEqual(w.value, 1.58113883)
        self.assertAlmostEqual(x.value, 0.3342179606)
        self.assertAlmostEqual(y.value, 1.165465304)

    def test_derivative(self):
        """Test _derivative method."""
        u = relentless.variable.DesignVariable(value=2.0)
        v = relentless.variable.DesignVariable(value=3.0)
        w = relentless.variable.Power(u,v)

        #test w.r.t. a
        dw = w._derivative('a')
        self.assertAlmostEqual(dw, 12.0)
        v.value = 0.0
        dw = w._derivative('a')
        self.assertAlmostEqual(dw, 0.0)
        v.value = 3.0

        #test w.r.t. b
        dw = w._derivative('b')
        self.assertAlmostEqual(dw, 5.545177444)
        u.value = 0.0
        dw = w._derivative('b')
        self.assertAlmostEqual(dw, 0.0)
        u.value = 2.0

        #test w.r.t. ~a,~b
        with self.assertRaises(ValueError):
            w._derivative('c')

        #test with negative a
        u.value = -2.0
        v.value = 2.5
        dw = w._derivative('b')
        self.assertEqual(dw, numpy.inf)

class test_ArithmeticMean(unittest.TestCase):
    """Unit tests for relentless.variable.ArithmeticMean"""

    def test_init(self):
        """Test creation with data."""
        #test variable dependent on 2 DesignVariables
        u = relentless.variable.DesignVariable(value=1.0)
        v = relentless.variable.DesignVariable(value=2.0)
        w = relentless.variable.ArithmeticMean(u, v)
        self.assertAlmostEqual(w.value, 1.5)
        self.assertCountEqual(w.params, ('a','b'))
        self.assertDictEqual({p:v for p,v in w.depends}, {'a':u,'b':v})

        #test variable dependent on 1 DesignVar, 1 DependentVar
        x = relentless.variable.ArithmeticMean(v,w)
        self.assertAlmostEqual(x.value, 1.75)
        self.assertCountEqual(x.params, ('a','b'))
        self.assertDictEqual({p:v for p,v in x.depends}, {'a':v,'b':w})

        #test variable dependent on 2 DependentVariables
        y = relentless.variable.ArithmeticMean(w,x)
        self.assertAlmostEqual(y.value, 1.625)
        self.assertCountEqual(y.params, ('a','b'))
        self.assertDictEqual({p:v for p,v in y.depends}, {'a':w,'b':x})

        #test same-object dependencies
        w = relentless.variable.ArithmeticMean(u, u)
        self.assertAlmostEqual(w.value, 1.0)
        self.assertCountEqual(w.params, ('a','b'))
        self.assertDictEqual({p:v for p,v in w.depends}, {'a':u,'b':u})

        #test scalar dependencies
        z = relentless.variable.ArithmeticMean(2.0, 1.0)
        self.assertAlmostEqual(z.value, 1.5)
        self.assertCountEqual(z.params, ('a','b'))
        self.assertDictEqual({p:v.value for p,v in z.depends}, {'a':2.0,'b':1.0})

    def test_value(self):
        """Test setting values."""
        #create "chained" dependent variables
        u = relentless.variable.DesignVariable(value=2.0, low=1.5, high=2.5)
        v = relentless.variable.DesignVariable(value=1.0, low=0.5, high=1.5)
        w = relentless.variable.ArithmeticMean(u, v)
        x = relentless.variable.ArithmeticMean(v, w)
        y = relentless.variable.ArithmeticMean(w, x)

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
        u = relentless.variable.DesignVariable(value=1.0)
        v = relentless.variable.DesignVariable(value=2.0)
        w = relentless.variable.ArithmeticMean(u, v)

        #test w.r.t. a
        dw = w._derivative('a')
        self.assertEqual(dw, 0.5)

        #test w.r.t. b
        dw = w._derivative('b')
        self.assertEqual(dw, 0.5)

        #test w.r.t. ~a,~b
        with self.assertRaises(ValueError):
            w._derivative('c')

        #change values and retest derivative
        u.value = 1.5
        v.value = 2.5
        dw = w._derivative('b')
        self.assertEqual(dw, 0.5)

class test_GeometricMean(unittest.TestCase):
    """Unit tests for relentless.variable.GeometricMean"""

    def test_init(self):
        """Test creation with data."""
        #test variable dependent on 2 DesignVariables
        u = relentless.variable.DesignVariable(value=1.0)
        v = relentless.variable.DesignVariable(value=16.0)
        w = relentless.variable.GeometricMean(u, v)
        self.assertAlmostEqual(w.value, 4.0)
        self.assertCountEqual(w.params, ('a','b'))
        self.assertDictEqual({p:v for p,v in w.depends}, {'a':u,'b':v})

        #test variable dependent on 1 DesignVar, 1 DependentVar
        x = relentless.variable.GeometricMean(v,w)
        self.assertAlmostEqual(x.value, 8.0)
        self.assertCountEqual(x.params, ('a','b'))
        self.assertDictEqual({p:v for p,v in x.depends}, {'a':v,'b':w})

        #test variable dependent on 2 DependentVariables
        y = relentless.variable.GeometricMean(w,x)
        self.assertAlmostEqual(y.value, numpy.sqrt(32.0))
        self.assertCountEqual(y.params, ('a','b'))
        self.assertDictEqual({p:v for p,v in y.depends}, {'a':w,'b':x})

        #test same-object dependencies
        w = relentless.variable.GeometricMean(v, v)
        self.assertAlmostEqual(w.value, 16.0)
        self.assertCountEqual(w.params, ('a','b'))
        self.assertDictEqual({p:v for p,v in w.depends}, {'a':v,'b':v})

        #test scalar dependencies
        z = relentless.variable.GeometricMean(16.0, 1.0)
        self.assertAlmostEqual(z.value, 4.0)
        self.assertCountEqual(z.params, ('a','b'))
        self.assertDictEqual({p:v.value for p,v in z.depends}, {'a':16.0,'b':1.0})

    def test_value(self):
        """Test setting values."""
        #create "chained" dependent variables
        u = relentless.variable.DesignVariable(value=2.0, low=1.5, high=2.5)
        v = relentless.variable.DesignVariable(value=1.0, low=0.5, high=1.5)
        w = relentless.variable.GeometricMean(u, v)
        x = relentless.variable.GeometricMean(v, w)
        y = relentless.variable.GeometricMean(w, x)

        self.assertEqual(u.value, 2.0)
        self.assertEqual(v.value, 1.0)
        self.assertAlmostEqual(w.value, numpy.sqrt(2.0))
        self.assertAlmostEqual(x.value, numpy.sqrt(numpy.sqrt(2.0)))
        self.assertAlmostEqual(y.value, numpy.sqrt(numpy.sqrt(numpy.sqrt(2.0))*numpy.sqrt(2.0)))

        #change value of u
        u.value = 2.6
        self.assertEqual(u.value, 2.5)
        self.assertEqual(v.value, 1.0)
        self.assertAlmostEqual(w.value, numpy.sqrt(2.5))
        self.assertAlmostEqual(x.value, numpy.sqrt(numpy.sqrt(2.5)))
        self.assertAlmostEqual(y.value, numpy.sqrt(numpy.sqrt(numpy.sqrt(2.5))*numpy.sqrt(2.5)))

        #change value of v
        v.value = 0.4
        self.assertEqual(u.value, 2.5)
        self.assertEqual(v.value, 0.5)
        self.assertAlmostEqual(w.value, numpy.sqrt(1.25))
        self.assertAlmostEqual(x.value, numpy.sqrt(0.5*numpy.sqrt(1.25)))
        self.assertAlmostEqual(y.value, numpy.sqrt(numpy.sqrt(0.5*numpy.sqrt(1.25))*numpy.sqrt(1.25)))

    def test_derivative(self):
        """Test _derivative method."""
        u = relentless.variable.DesignVariable(value=1.0)
        v = relentless.variable.DesignVariable(value=2.0)
        w = relentless.variable.GeometricMean(u, v)

        #test w.r.t. a
        dw = w._derivative('a')
        self.assertAlmostEqual(dw, 0.5*numpy.sqrt(2.0))

        #test w.r.t. b
        dw = w._derivative('b')
        self.assertAlmostEqual(dw, 0.5*numpy.sqrt(0.5))

        #test w.r.t. ~a,~b
        with self.assertRaises(ValueError):
            w._derivative('c')

        #test with 0 denominator
        u.value = 0.0
        dw = w._derivative('a')
        self.assertEqual(dw, numpy.inf)

        v.value = 0.0
        dw = w._derivative('b')
        self.assertEqual(dw, numpy.inf)

if __name__ == '__main__':
    unittest.main()
