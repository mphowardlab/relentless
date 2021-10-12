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
.. autoclass:: ArithmeticMean
.. autoclass:: GeometricMean

.. autoclass:: Variable
    :members: value
.. autoclass:: IndependentVariable
    :members:
.. autoclass:: DependentVariable
    :members:
    :private-members: _derivative
.. autoclass:: UnaryOperator
    :members:
.. autoclass:: BinaryOperator
    :members:

"""
import abc
from enum import Enum

import networkx
import numpy

class Variable(abc.ABC):
    """Abstract base class for a variable.

    A variable represents a single scalar quantity. The :attr:`value` of the
    variable is an abstract property that must be implemented. Most users should
    not inherit from this variable directly but should make use of the more
    flexible :class:`IndependentVariable` or :class:`DependentVariable` types.

    """
    @property
    @abc.abstractmethod
    def value(self):
        """float: Value of the variable."""
        pass

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
        >>> print(x.value)
        3.0

    The value of an independent variable can be changed::

        >>> x.value = -1.0
        >>> print(x.value)
        -1.0

    """
    def __init__(self, value):
        self._value = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

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

    def __init__(self, value, const=False, low=None, high=None):
        super().__init__(value=value)
        self.const = const
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
        if not isinstance(value,(float,int)):
            raise ValueError('DesignVariable must be a float or int')
        self._value, self._state = self.clamp(value)

    @property
    def low(self):
        """float: Lower bound on value."""
        return self._low

    @low.setter
    def low(self, low):
        if low is not None and not isinstance(low, (float,int)):
            raise ValueError('Low bound must be a float or int')
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
            raise ValueError('High bound must be a float or int')
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
        attrs = {}
        for d in vardicts:
            attrs.update(d)
        attrs.update(**kwvars)

        if len(attrs) == 0:
            raise AttributeError('No attributes specified for DependentVariable.')

        for k,v in attrs.items():
            super().__setattr__(k,self._make_variable(v))
        self._params = tuple(attrs.keys())

    def __setattr__(self, name, value):
        #Sets the value of the variable on which the specified DependentVariable depends.
        if name != '_params' and name in self.params:
            value = self._make_variable(value)
        super().__setattr__(name,value)

    @property
    def depends(self):
        """Dependency generator.

        Generator expression for all the dependent attributes and variables for
        this object.

        Yields
        ------
        :class:`str`
            The name of the attribute.
        :class:`Variable`
            The parameter variable object.

        """
        for p in self.params:
            yield p,getattr(self,p)

    @property
    def params(self):
        """tuple[str]: Names of all dependencies."""
        return self._params

    def dependency_graph(self):
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

        Returns
        -------
        :class:`networkx.DiGraph`
            Directed dependency graph.

        """
        # discover all variables (nodes) by quasi-depth first search
        g = networkx.DiGraph()
        stack = [self]
        visited = set()
        while stack:
            a = stack.pop()
            if a not in visited:
                g.add_node(a)
                visited.add(a)
                if isinstance(a, DependentVariable):
                    for p,b in a.depends:
                        stack.append(b)
                        # add new edge for p if not already present, otherwise append to params list
                        if (a,b) not in g.edges:
                            g.add_edge(a,b,params=[p])
                        else:
                            g.edges[a,b]['params'].append(p)

        return g

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
        # if var is this variable, the derivative is trivially 1.0
        if var is self:
            return 1.0

        # get dependency graph
        g = self._assert_acyclic(self.dependency_graph())

        # if var is not in the graph, then its derivative is trivially 0.0
        if var not in g:
            return 0.0

        # add sum of parameter derivatives to edges between objects
        for a,b,params in g.edges.data('params'):
            g.edges[a,b]['deriv'] = numpy.sum([a._derivative(p) for p in params])

        # compute chain rule along all paths to the variable
        deriv = 0.
        paths = networkx.all_simple_paths(g, source=self, target=var)
        for path in map(networkx.utils.pairwise, paths):
            deriv += numpy.prod([g.edges[edge]['deriv'] for edge in path])
        return deriv

    @abc.abstractmethod
    def _derivative(self, param):
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

    @classmethod
    def _make_variable(cls, v):
        # cast a scalar into an independent variable
        if numpy.isscalar(v):
            v = IndependentVariable(value=v)
        if not isinstance(v, Variable):
            raise TypeError('Dependent variables can only depend on scalars or other variables.')
        return v

    @classmethod
    def _assert_acyclic(cls, g):
        # confirm dependency graph is free of cycles
        if not networkx.is_directed_acyclic_graph(g):
            raise RuntimeError('DependentVariable has circular dependencies.')
        return g

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
    @property
    def value(self):
        return self.a.value

    def _derivative(self, param):
        if param == 'a':
            return 1.0
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
    @property
    def value(self):
        return 0.5*(self.a.value+self.b.value)

    def _derivative(self, param):
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
    @property
    def value(self):
        return numpy.sqrt(self.a.value*self.b.value)

    def _derivative(self, param):
        if param == 'a':
            try:
                return 0.5*numpy.sqrt(self.b.value/self.a.value)
            except ZeroDivisionError:
                return numpy.inf
        elif param == 'b':
            try:
                return 0.5*numpy.sqrt(self.a.value/self.b.value)
            except ZeroDivisionError:
                return numpy.inf
        else:
            raise ValueError('Unknown parameter')
