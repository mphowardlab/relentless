"""
Variables
=========

A :class:`Variable` represents a single scalar quantity of interest. We can
distinguish between two types of variables: an :class:`IndependentVariable`, which
does not depend on other variables, and a :class:`DependentVariable`, which does.
Typically, the :class:`IndependentVariable` is a quantity that will be held
constant or adjusted, while a :class:`DependentVariable` is a function of other
these values.

The :class:`DesignVariable` is a very useful :class:`IndependentVariable` for
doing design (optimization), as it allows you to specify a value, its bounds,
and whether it is adjustable. Other classes within :mod:`relentless` that
accept variables as values will often automatically detect the adjustable
:class:`DesignVariable` values for use in design problems, ignoring the other
variables as adjustable.

The following independent and dependent variables have been implemented:

.. autosummary::
    :nosignatures:

    DesignVariable
    SameAs
    Negation
    Sum
    Difference
    Product
    Quotient
    Power
    ArithmeticMean
    GeometricMean

.. rubric:: Developer notes

To implement your own variables, create a class that derives from one of the
following classes (typically :class:`DependentVariable`, :class:`UnaryOperator`,
or :class:`BinaryOperator`) and define the required properties and methods.

.. autosummary::
    :nosignatures:

    Variable
    IndependentVariable
    DependentVariable
    UnaryOperator
    BinaryOperator

.. autoclass:: DesignVariable
    :members:
.. autoclass:: SameAs
.. autoclass:: Negation
.. autoclass:: Sum
.. autoclass:: Difference
.. autoclass:: Product
.. autoclass:: Quotient
.. autoclass:: Power
.. autoclass:: ArithmeticMean
.. autoclass:: GeometricMean

.. autoclass:: Variable
    :members: value
.. autoclass:: IndependentVariable
    :members:
.. autoclass:: DependentVariable
    :members:
    :private-members: compute, compute_derivative
.. autoclass:: UnaryOperator
    :members:
.. autoclass:: BinaryOperator
    :members:

"""
import abc
from enum import Enum

import networkx
import numpy

class VariableGraph:
    """Construct a directed graph of dependencies.

    The graph nodes are all the :class:`Variable` objects on which this
    :class:`DependentVariable` is dependent, and the graph edges represent
    dependencies as connections between variables and carry the name of the
    parameter as a "weight". The edge weight is a list of named parameters,
    so an object can depend on the same :class:`Variable` multiple times;
    these multiple dependencies are represented as one edge with multiple
    parameters listed.

    The graph is constructed using a quasi depth-first search. In order to
    perform certain calculations, the directed graph should be acyclic.

    """
    def __init__(self):
        self._graph = networkx.MultiDiGraph()
        self._is_acyclic = None
        self._constants = {}

    def ensure_variable(self, x):
        if isinstance(x, (float, int)):
            if x in self._constants:
                x_ = self._constants[x]
            else:
                x_ = ConstantVariable(x)
                # cache auto-created constants to reduce number of variables
                self._constants[x] = x_
        else:
            x_ = x

        # confirm x has right type
        if not isinstance(x_, Variable):
            raise TypeError('Type cannot be coerced to a variable.')

        return x_

    def add_variable(self, x):
        x = self.ensure_variable(x)
        if x not in self._graph.nodes:
            self._graph.add_node(x)
            # invalidate acyclic cache value since variables have changed
            self._is_acyclic = None
        return x

    def set_dependencies(self, x, depends):
        if x not in self._graph.nodes:
            raise ValueError('Variable has not been added to graph')
        elif not isinstance(x, DependentVariable):
            raise TypeError('Dependencies can only be set for DependentVariable')

        # remove any old edges
        self._graph.remove_edges_from(self._graph.edges(x))

        # set new edges
        for p, y in depends.items():
            y = self.add_variable(y)
            self._graph.add_edge(x, y, depend=str(p))

        # invalidate acyclic cache value since edges have changed
        self._is_acyclic = None

    def update_variable(self, x):
        if x not in self._graph.nodes:
            raise ValueError('Variable has not been added to graph')
        elif isinstance(x, DependentVariable):
            raise TypeError('Cannot update value of dependent variable directly')
        for y in networkx.dfs_preorder_nodes(self._graph.reverse(), source=x):
            if isinstance(y, DependentVariable):
                y._recompute = True

    def evaluate(self, x):
        if isinstance(x, DependentVariable):
            if not self.is_acyclic:
                raise RuntimeError('Circuluar variable dependency in graph')

            # evaluate the nodes, bottom up
            for y in networkx.dfs_postorder_nodes(self._graph, source=x):
                if isinstance(y, DependentVariable):
                    depends = self._graph.edges(y, data='depend')
                    # access _value directly to avoid evaluations
                    y._value = y.compute(**{p: z.value for _,z,p in depends})
                    y._recompute = False

    def evaluate_derivative(self, f, x):
        if f is x:
            return 1.0

        if not self.is_acyclic:
            raise RuntimeError('Circuluar variable dependency in graph')

        # check that a path exists before searching
        if not networkx.has_path(self._graph, source=f, target=x):
            return 0.0

        # ensure graph values are up-to-date for f, x will get updated too
        self.evaluate(f)

        # compute chain rule along all paths to the variable
        deriv = 0.
        for path in networkx.all_simple_edge_paths(self._graph, source=f, target=x):
            path_deriv = 1.
            for edge in path:
                param = self._graph.edges[edge]['depend']
                depends = self._graph.edges(edge[0], data='depend')
                path_deriv *= edge[0].compute_derivative(param, **{p: y.value for _,y,p in depends})
            deriv += path_deriv
        return deriv

    @property
    def variables(self):
        return self._graph.nodes

    @property
    def is_acyclic(self):
        if self._is_acyclic is None:
            self._is_acyclic = networkx.is_directed_acyclic_graph(self._graph)
        return self._is_acyclic
graph = VariableGraph()

class Variable(abc.ABC):
    """Abstract base class for a variable.

    A variable represents a single scalar quantity. The :attr:`value` of the
    variable is an abstract property that must be implemented. Most users should
    not inherit from this variable directly but should make use of the more
    flexible :class:`IndependentVariable` or :class:`DependentVariable` types.

    Examples
    --------
    Create variables with specified value::

        x = relentless.variable.IndependentVariable(value=2.0)

    If the value is numerical, arithmetic operations are available::

        >>> print(x+2.0)
        4.0
        >>> print(5.0-x)
        1.0
        >>> print(2.0*x)
        2.0
        >>> print(x/4.0)
        0.5
        >>> print(x**2)
        0.25
        >>> print(-x)
        -0.25

    """
    count = 0
    names = set()

    def __init__(self, name=None):
        self.id = Variable.count
        if name is None:
            name = 'x[{}]'.format(self.id)
        if name in self.names:
            raise ValueError('Variable name already used')
        else:
            self.names.add(name)
            self.name = name
        Variable.count += 1
        graph.add_variable(x=self)

    @property
    @abc.abstractmethod
    def value(self):
        """float: Value of the variable."""
        pass

    def __add__(self, val):
        """Addition of two variables, or of a variable and a scalar."""
        return Sum(self, val)

    def __radd__(self, val):
        """Addition of a scalar and a variable."""
        return Sum(val, self)

    def __sub__(self, val):
        """Subtraction of two variables, or of a variable and a scalar."""
        return Difference(self, val)

    def __rsub__(self, val):
        """Subtraction of a scalar and a variable."""
        return Difference(val, self)

    def __mul__(self, val):
        """Multiplication of two variables, or of a variable and a scalar."""
        return Product(self, val)

    def __rmul__(self, val):
        """Multiplication of a scalar and a variable."""
        return Product(val, self)

    def __truediv__(self, val):
        """Division of two variables, or of a variable by a scalar."""
        return Quotient(self, val)

    def __rtruediv__(self, val):
        """Division of a scalar by a variable."""
        return Quotient(val, self)

    def __pow__(self, val):
        """Exponentiation of two variables, or of a variable with a scalar."""
        return Power(self, val)

    def __neg__(self):
        """Negation of a variable."""
        return Negation(self)

    def __str__(self):
        return str(self.value)

class ConstantVariable(Variable):
    def __init__(self, value, name=None):
        if not isinstance(value, (float, int)):
            raise TypeError('Constant values are only floats or ints')
        super().__init__(name=name)
        self._value = value

    @property
    def value(self):
        return self._value

class IndependentVariable(Variable):
    """Independent quantity.

    Parameters
    ----------
    value : float
        Initial value.

    Examples
    --------
    Create an independent variable::

        >>> x = relentless.variable.IndependentVariable(3.0)
        >>> print(x)
        3.0

    The value of an independent variable can be changed::

        >>> x.value = -1.0
        >>> print(x)
        -1.0

    Perform in-place arithmetic operations::

        >>> x += 4.0
        >>> print(x)
        3.0
        >>> x -= 1.0
        >>> print(x)
        2.0
        >>> x *= 0.5
        >>> print(x)
        1.0
        >>> x /= 2.0
        >>> print(x)
        0.5

    """
    def __init__(self, value, name=None):
        if not isinstance(value, (float, int)):
            raise TypeError('Independent variables are only float or int')
        super().__init__(name=name)
        self._value = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if not isinstance(value, (float, int)):
            raise TypeError('Independent variables are only float or int')
        self._value = value
        graph.update_variable(self)

    def __iadd__(self, val):
        """In-place addition of a variable with a scalar."""
        if isinstance(val, Variable):
            raise TypeError('Variables are not allowed to operate in-place on another Variable')
        self.value += val
        return self

    def __isub__(self, val):
        """In-place subtraction of a variable with a scalar."""
        if isinstance(val, Variable):
            raise TypeError('Variables are not allowed to operate in-place on another Variable')
        self.value -= val
        return self

    def __imul__(self, val):
        """In-place multiplication of a variable by a scalar."""
        if isinstance(val, Variable):
            raise TypeError('Variables are not allowed to operate in-place on another Variable')
        self.value *= val
        return self

    def __itruediv__(self, val):
        """In-place division of a variable by a scalar."""
        if isinstance(val, Variable):
            raise TypeError('Variables are not allowed to operate in-place on another Variable')
        self.value /= val
        return self

class DesignVariable(IndependentVariable):
    """Constrained independent variable.

    A design variable is a quantity that is meant to be adjusted, e.g., by
    optimization. Optional box constraints (lower and upper bounds) can be
    specified for the variable. When set, these bounds will be respected and
    an internal state will track whether the requested quantity was within or
    outside these bounds. This is useful for performing constrained
    optimization and for ensuring physical quantities have meaningful values
    (e.g., lengths should be positive).

    Parameters
    ----------
    value : float or int
        Value of the variable.
    const : bool
        If ``False``, the variable can be optimized; otherwise, it should be
        treated as a constant (defaults to ``False``).
    low : float or None
        Lower bound for the variable (``None`` means no lower bound).
    high : float or None
        Upper bound for the variable (``None`` means no upper bound).

    Examples
    --------
    A variable with a lower bound::

        >>> v = relentless.variable.DesignVariable(value=1.0, low=0.0)
        >>> v.value
        1.0
        >>> v.isfree()
        True
        >>> v.atlow()
        False

    Bounds are respected and noted when setting values::

        >>> v.value = -1.0
        >>> v.value
        0.0
        >>> v.isfree()
        False
        >>> v.atlow()
        True

    """

    class State(Enum):
        """Constrained state of the variable.

        Attributes
        ----------
        FREE : int
            Variable is not constrained or within bounds.
        LOW : int
            Variable is constrained to lower bound.
        HIGH : int
            Variable is constrained to upper bound.

        """
        FREE = 0
        LOW = 1
        HIGH = 2

    def __init__(self, value, name=None, low=None, high=None):
        if not isinstance(value, (float, int)):
            raise TypeError('Design variables are only float or int')
        super().__init__(name=name, value=value)
        self.low = low
        self.high = high

    def clamp(self, value):
        """Clamp value within the bounds of the variable.

        Parameters
        ----------
        value : float
            Value to clamp.

        Returns
        -------
        v : float
            Constrained value.
        b : :class:`State`
            Constrained state of the value.

        """
        if self.low is not None and value <= self.low:
            v = self.low
            b = DesignVariable.State.LOW
        elif self.high is not None and value >= self.high:
            v = self.high
            b = DesignVariable.State.HIGH
        else:
            v = value
            b = DesignVariable.State.FREE

        return v,b

    @IndependentVariable.value.setter
    def value(self, value):
        if not isinstance(value, (float, int)):
            raise TypeError('Design variables are only float or int')
        self._value, self._state = self.clamp(value)
        graph.update_variable(self)

    @property
    def low(self):
        """float: Lower bound on value."""
        return self._low

    @low.setter
    def low(self, low):
        if low is not None and not isinstance(low, (float,int)):
            raise TypeError('Low bound must be a float or int')
        try:
            if low is None or self._high is None:
                self._low = low
            elif low < self._high:
                self._low = low
            else:
                raise ValueError('The low bound must be less than the high bound')
        except (AttributeError,TypeError):
            self._low = low

        try:
            self._value, self._state = self.clamp(self._value)
        except AttributeError:
            pass

    @property
    def high(self):
        """float: Upper bound on value."""
        return self._high

    @high.setter
    def high(self, high):
        if high is not None and not isinstance(high, (float,int)):
            raise TypeError('High bound must be a float or int')
        try:
            if high is None or self._high is None:
                self._high = high
            elif high > self._low:
                self._high = high
            else:
                raise ValueError('The high bound must be greater than the low bound')
        except (AttributeError,TypeError):
            self._high = high

        try:
            self._value, self._state = self.clamp(self._value)
        except AttributeError:
            pass

    @property
    def state(self):
        """:class:`State`: Constraint state of the variable."""
        return self._state

    def isfree(self):
        """Check if variable is not constrained by bounds.

        Returns
        -------
        bool
            True if the variable is not constrained.

        """
        return self.state is self.State.FREE

    def atlow(self):
        """Check if variable is constrained at the lower bound.

        Returns
        -------
        bool
            True if the variable is constrained at the lower bound.

        """
        return self.state is self.State.LOW

    def athigh(self):
        """Check if variable is constrained at the upper bound.

        Returns
        -------
        bool
            True if the variable is constrained at the lower bound.

        """
        return self.state is self.State.HIGH

class DependentVariable(Variable):
    """Abstract base class for a variable that depends on other values.

    A dependent variable is composed from other variables. This is an abstract
    base class for any such variable. In addition to the :attr:`~relentless.variable.Variable.value`
    property of the :class:`Variable`, a :class:`DependentVariable` must also define a
    :meth:`_derivative` method that defines its partial derivative with respect
    to its dependencies.

    The names of the dependencies of the variable are automatically deduced
    by the base constructor using dictionary keys and keyword arguments and
    assigned as attributes of the object. For example::

        class NewVariable(relentless.variable.DependentVariable):
            def __init__(self, a, b):
                super().__init__({'a': a}, b=b)

    will create a ``NewVariable`` with two dependent attributes ``a`` and ``b``.

    Dependencies are typically other :class:`Variable` objects, but they can also
    be set to scalar values. In these case, the scalar is converted to an
    :class:`IndependentVariable` first.

    Parameters
    ----------
    vardicts : dict
        Dependencies as entries in an arbitrary number of dictionaries.
    kwvars : kwargs
        Dependencies as an arbitrary number of keyword arguments.

    """
    def __init__(self, *vardicts, **kwvars):
        super().__init__()

        # combine list of dependencies
        depends = {}
        for d in vardicts:
            depends.update(d)
        depends.update(**kwvars)
        assert len(depends) != 0
        self._params = tuple(depends.keys())

        graph.set_dependencies(x=self, depends=depends)
        self._value = None
        self._recompute = True

    @property
    def value(self):
        if self._recompute:
            graph.evaluate(self)
        return self._value

    @property
    def params(self):
        return self._params

    def derivative(self, var):
        """Calculate derivative with respect to a :class:`Variable`.

        The derivative is evaluated using the standard chain rule.

        Parameters
        ----------
        var : :class:`Variable`
            Variable with respect to which to take the derivative.

        Returns
        -------
        float
            The calculated derivative.

        Raises
        ------
        RuntimeError
            If this :class:`DependentVariable` has any circular dependencies.

        """
        if self._recompute:
            graph.evaluate(self)
        return graph.evaluate_derivative(f=self, x=var)

    @abc.abstractmethod
    def compute(self):
        pass

    @abc.abstractmethod
    def compute_derivative(self, param):
        """Implementation of the derivative.

        This method should implement the partial derivative with respect
        to the named dependency ``param`` given the current value of this
        variable.

        Parameters
        ----------
        param : str
            Name of the dependency.

        """
        pass

class UnaryOperator(DependentVariable):
    """Abstract base class for a value that depends on one variable.

    Deriving classes still need to implement the :attr:`~relentless.variable.Variable.value`
    and :meth:`~relentless.variable.DependentVariable._derivative` methods.
    :class:`UnaryOperator` is a convenience class implementing the constructor
    of a function that depends on one value, :math:`f(a)`.

    Parameters
    ----------
    a : :class:`Variable`
        The :class:`Variable` this value depends on.

    """
    def __init__(self, a):
        super().__init__(a=a)

class SameAs(UnaryOperator):
    """Copy a value.

    The value of this variable will mirror the value of ``a``. It is equivalent
    to using ``a`` directly::

        >>> a = relentless.variable.IndependentVariable(2.0)
        >>> b = relentless.variable.SameAs(a)
        >>> print(b.value)
        2.0
        >>> a.value = 3.0
        >>> print(b.value)
        3.0

    Parameters
    ----------
    a : :class:`Variable`
        Variable to copy.

    """
    def compute(self, a):
        return a

    def compute_derivative(self, param, a):
        if param == 'a':
            return 1.0
        else:
            raise ValueError('Unknown parameter')

class Negation(UnaryOperator):
    r"""Takes the additive inverse of a value.

    Given a value :math:`a`, this returns :math:`-a`

    Parameters
    ----------
    a : :class:`Variable`
        The value.

    """
    def compute(self, a):
        return -a

    def compute_derivative(self, param, a):
        if param == 'a':
            return -1.0
        else:
            raise ValueError('Unknown parameter')

class BinaryOperator(DependentVariable):
    """Abstract base class for a value that depends on two variables.

    Deriving classes still need to implement the :attr:`~relentless.variable.Variable.value`
    and :meth:`~relentless.variable.DependentVariable._derivative` methods.
    :class:`BinaryOperator` is a convenience class implementing the constructor of
    a function that depends on two values, :math:`f(a,b)`.

    Parameters
    ----------
    a : :class:`Variable`
        The first :class:`Variable` this value depends on.
    b : :class:`Variable`
        The second :class:`Variable` this value depends on.

    """
    def __init__(self, a, b):
        super().__init__(a=a, b=b)

class Sum(BinaryOperator):
    r"""Sum of two values.

    Given two values :math:`a` and :math:`b`, this returns :math:`a+b`

    Parameters
    ----------
    a : :class:`Variable`
        First value.
    b : :class:`Variable`
        Second value.

    """
    def compute(self, a, b):
        return a+b

    def compute_derivative(self, param, a, b):
        if param == 'a':
            return 1.0
        elif param == 'b':
            return 1.0
        else:
            raise ValueError('Unknown parameter')

class Difference(BinaryOperator):
    r"""Difference of two values.

    Given two values :math:`a` and :math:`b`, this returns :math:`a-b`

    Parameters
    ----------
    a : :class:`Variable`
        First value.
    b : :class:`Variable`
        Second value.

    """
    def compute(self, a, b):
        return a-b

    def compute_derivative(self, param, a, b):
        if param == 'a':
            return 1.0
        elif param == 'b':
            return -1.0
        else:
            raise ValueError('Unknown parameter')

class Product(BinaryOperator):
    r"""Product of two values.

    Given two values :math:`a` and :math:`b`, this returns :math:`ab`

    Parameters
    ----------
    a : :class:`Variable`
        First value.
    b : :class:`Variable`
        Second value.

    """
    def compute(self, a, b):
        return a*b

    def compute_derivative(self, param, a, b):
        if param == 'a':
            return b
        elif param == 'b':
            return a
        else:
            raise ValueError('Unknown parameter')

class Quotient(BinaryOperator):
    r"""Quotient of two values.

    Given two values :math:`a` and :math:`b`, this returns :math:`\frac{a}{b}`

    Parameters
    ----------
    a : :class:`Variable`
        First value.
    b : :class:`Variable`
        Second value.

    """
    def compute(self, a, b):
        if b != 0:
            return a/b
        else:
            return numpy.nan

    def compute_derivative(self, param, a, b):
        if param == 'a':
            if b != 0:
                return 1.0/b
            else:
                return numpy.nan
        elif param == 'b':
            if b != 0:
                return -a/numpy.power(b, 2)
            else:
                return numpy.nan
        else:
            raise ValueError('Unknown parameter')

class Power(BinaryOperator):
    r"""Takes a value to a power.

    Given two values :math:`a` and :math:`b`, this returns :math:`a^b`

    Parameters
    ----------
    a : :class:`Variable`
        First value.
    b : :class:`Variable`
        Second value.

    """
    def compute(self, a, b):
        return numpy.power(a, b)

    def compute_derivative(self, param, a, b):
        if param == 'a':
            if b == 0:
                return 0.0
            return b*numpy.power(a, b-1)
        elif param == 'b':
            if a == 0:
                return 0.0
            else:
                return numpy.log(a)*numpy.power(a, b)
        else:
            raise ValueError('Unknown parameter')

class ArithmeticMean(BinaryOperator):
    r"""Arithmetic mean of two values.

    The arithmetic mean :math:`v` of two values :math:`a` and :math:`b` is:

    .. math::

        v = \frac{a+b}{2}

    This variable may be useful for implementing mixing rules.

    Parameters
    ----------
    a : :class:`Variable`
        First value.
    b : :class:`Variable`
        Second value.

    """
    def compute(self, a, b):
        return 0.5*(a+b)

    def compute_derivative(self, param, a, b):
        if param == 'a':
            return 0.5
        elif param == 'b':
            return 0.5
        else:
            raise ValueError('Unknown parameter')

class GeometricMean(BinaryOperator):
    r"""Geometric mean of two values.

    The geometric mean :math:`v` of two values :math:`a` and :math:`b` is:

    .. math::

        v = \sqrt{a b}

    This variable may be useful for implementing mixing rules.

    Parameters
    ----------
    a : :class:`Variable`
        First value.
    b : :class:`Variable`
        Second value.

    """
    def compute(self, a, b):
        return numpy.sqrt(a*b)

    def compute_derivative(self, param, a, b):
        if param == 'a':
            if a != 0:
                return 0.5*numpy.sqrt(b/a)
            else:
                return numpy.nan
        elif param == 'b':
            if b != 0:
                return 0.5*numpy.sqrt(a/b)
            else:
                return numpy.nan
        else:
            raise ValueError('Unknown parameter')
