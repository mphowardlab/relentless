__all__ = ['Variable','IndependentVariable','DependentVariable',
           'DesignVariable',
           'UnaryOperator','SameAs',
           'BinaryOperator','ArithmeticMean','GeometricMean'
          ]

import abc
from enum import Enum

import networkx as nx
import numpy as np

class Variable(abc.ABC):
    """Abstract base class for any variable.

    At minimum, a variable must provide the interface to get its value as a property.

    """
    @property
    @abc.abstractmethod
    def value(self):
        pass

class IndependentVariable(Variable):
    """Variable representing one independent quantity.

    Parameters
    ----------
    value : scalar or array
        Value of the variable.

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
    """Designable variable.

    Represents a quantity that optionally takes lower and upper bounds.
    When the value of the quantity is set, these bounds will be respected and
    an internal state will track whether the requested quantity was within or
    outside these bounds. This is useful for performing constrained
    optimization and for ensuring physical quantities have meaningful values
    (e.g., lengths should be positive).

    Parameters
    ----------
    value : float or int
        Value of the variable.
    const : bool
        If `False`, the variable can be optimized; otherwise, it is treated as
        a constant in the optimization (defaults to `False`).
    low : float or None
        Lower bound for the variable (`None` means no lower bound).
    high : float or None
        Upper bound for the variable (`None` means no upper bound).

    Examples
    --------
    A variable with a lower bound::

        >>> v = DesignVariable(value=1.0, low=0.0)
        >>> v.value
        1.0
        >>> v.isfree()
        True
        >>> v.atlow()
        False

    Bounds are respected and noted::

        >>> v.value = -1.0
        >>> v.value
        0.0
        >>> v.isfree()
        False
        >>> v.atlow()
        True

    """

    class State(Enum):
        """State of the DesignVariable.

        Attributes
         ----------
        FREE : int
            Value if the variable is unconstrained or within the defined bounds.
        LOW : int
            Value if the variable is clamped to the lower bound.
        HIGH : int
            Value if the variable is clamped to the upper bound.

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
        """Clamps a value within the bounds.

        Parameters
        ----------
        value : float
            Value to clamp within bounds.

        Returns
        -------
        v : float
            The clamped value.
        b : :py:class:`DesignVariable.State`
            The state of the variable.

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
        """float: The low bound of the variable"""
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
        except AttributeError:
            self._low = low

        try:
            self._value, self._state = self.clamp(self._value)
        except AttributeError:
            pass

    @property
    def high(self):
        """float: The high bound of the variable"""
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
        except AttributeError:
            self._high = high

        try:
            self._value, self._state = self.clamp(self._value)
        except AttributeError:
            pass

    @property
    def state(self):
        """:py:class:`DesignVariable.State`: The state of the variable"""
        return self._state

    def isfree(self):
        """Confirms if the variable is unconstrained or within the bounds.

        Returns
        -------
        bool
            True if the variable is unconstrained or within the bounds, False otherwise.

        """
        return self.state is DesignVariable.State.FREE

    def atlow(self):
        """Confirms if the variable is at the lower bound.

        Returns
        -------
        bool
            True if the variable is at the lower bound, False otherwise.

        """
        return self.state is DesignVariable.State.LOW

    def athigh(self):
        """Confirms if the variable is at the upper bound.

        Returns
        -------
        bool
            True if the variable is at the upper bound, False otherwise.

        """
        return self.state is DesignVariable.State.HIGH

class DependentVariable(Variable):
    """Abstract base class for a variable that depends on other variables.

    A dependent variable is composed from other variables. In addition to the
    value property, it must also supply a derivative method with respect to
    any of its dependent variables.

    The dependent variable attribute can also be set as a scalar value, but during
    construction the value is casted to a :py:class:`IndepedentVariable`.

    The number of dependencies of the variable are locked at construction, but
    their values can be modified.

    Parameters
    ----------
    vardicts : dict
        Attributes for the variable on which the DependentVariable depends (positional argument).
    kwvars : kwargs
        Attributes for the variable on which the DependentVariable depends (keyword arguments).

    """
    def __init__(self, *vardicts, **kwvars):
        attrs = {}
        for d in vardicts:
            attrs.update(d)
        attrs.update(**kwvars)

        if len(attrs) == 0:
            raise AttributeError('No attributes specified for DependentVariable.')

        for k,v in attrs.items():
            if np.isscalar(v):
                v = IndependentVariable(value=v)
            super().__setattr__(k,self._assert_variable(v))
        self._params = tuple(attrs.keys())

    def __setattr__(self, name, value):
        #Sets the value of the variable on which the specified DependentVariable depends.
        if name != '_params' and name in self.params:
            value = self._assert_variable(value)
        super().__setattr__(name,value)

    @property
    def depends(self):
        """Generates all the variables and the parameter names on which the specified DependentVariable depends.

        Yields
        ------
        str
            The parameter variable name.
        :py:class:`DependentVariable`
            The parameter variable object.

        """
        for p in self.params:
            yield p,getattr(self,p)

    @property
    def params(self):
        """tuple: Parameter names for all dependencies of the specified DependentVariable."""
        return self._params

    def dependency_graph(self):
        """Constructs a networkx graph of all variable objects on which the specified DependentVariable depends.

        The graph nodes are all variable objects which DependentVariable are dependent on.
        The directed graph edges represent dependencies as connections between variables.
        The graph can also take into account the dependencies of a DependentVariable which has
        multiple parameters as the same object.

        Returns
        -------
        :py:class:`networkx.DiGraph`
            The graph of all dependencies for this DependentVariable.

        """
        # construct graph of variables

        # discover all variables (nodes) by quasi-depth first search
        g = nx.DiGraph()
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
        """Calculates the derivative of a DependentVariable object
        with respect to another Variable object.

        Parameters
        ----------
        var : :py:class:`Variable`
            The variable with respect to which to take the derivative.

        Returns
        -------
        float
            The calculated derivative value.

        Raises
        ------
        RuntimeError
            If the specified DependentVariable object has any circular dependencies.

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
            g.edges[a,b]['deriv'] = np.sum([a._derivative(p) for p in params])

        # compute chain rule along all paths to the variable
        deriv = 0.
        paths = nx.all_simple_paths(g, source=self, target=var)
        for path in map(nx.utils.pairwise, paths):
            deriv += np.prod([g.edges[edge]['deriv'] for edge in path])
        return deriv

    @abc.abstractmethod
    def _derivative(self, param):
        pass

    @classmethod
    def _assert_variable(cls, v):
        #Checks if the dependent variable depends on another variable.
        if not isinstance(v, Variable):
            raise TypeError('Dependent variables can only depend on other variables.')
        return v

    @classmethod
    def _assert_acyclic(cls, g):
        # confirm dependency graph is free of cycles
        if not nx.is_directed_acyclic_graph(g):
            raise RuntimeError('DependentVariable has circular dependencies.')
        return g

class UnaryOperator(DependentVariable):
    """Abstract base class for a value that depends on one :py:class:`Variable`.

    Parameters
    ----------
    a : :py:class:`Variable`
        The variable object on which the UnaryOperator depends.

    """
    def __init__(self, a):
        super().__init__(a=a)

class SameAs(UnaryOperator):
    """Unary operator for copying a :py:class:`Variable`.

    The SameAs object has the same value as the object it depends on.

    Parameters
    ----------
    a : :py:class:`Variable`
        The variable object on which the UnaryOperator depends.

    """
    @property
    def value(self):
        return self.a.value

    def _derivative(self, param):
        """Calculates the derivative of the specified SameAs object with respect to its parameter.

        Parameters
        ----------
        param : str
            The parameter with respect to which to take the derivative.
            (Can only be 'a').

        Returns
        -------
        float
            The calculated derivative value.

        Raises
        ------
        ValueError
            If the parameter argument is not 'a'.

        """
        if param == 'a':
            return 1.0
        else:
            raise ValueError('Unknown parameter')

class BinaryOperator(DependentVariable):
    """Abstract base class for a value that depends on two :py:class:`Variable`s.

    Parameters
    ----------
    a : :py:class:`Variable`
        The first variable object on which the BinaryOperator depends.
    b : :py:class:`Variable`
        The second variable object on which the BinaryOperator depends.

    """
    def __init__(self, a, b):
        super().__init__(a=a, b=b)

class ArithmeticMean(BinaryOperator):
    r"""Binary operator based on arithmetic mean.

    The ArithmeticMean object akes the arithmetic mean of two :py:class:`Variable`s
    as shown below:

    .. math::

        v = \frac{1}{2}(a+b)

    This class can also be used to implement mixing rules.

    Parameters
    ----------
    a : :py:class:`Variable`
        The first parameter of the ArithmeticMean.
    b : :py:class:`Variable`
        The second parameter of the ArithmeticMean.

    """
    @property
    def value(self):
        return 0.5*(self.a.value+self.b.value)

    def _derivative(self, param):
        """Calculates the derivative of the specified ArithmeticMean object with respect to its parameters.

        Parameters
        ----------
        param : str
            The parameter with respect to which to take the derivative.
            (Can only be 'a' or 'b').

        Returns
        -------
        float
            The calculated derivative value.

        Raises
        ------
        ValueError
            If the parameter argument is not 'a' or 'b'.

        """
        if param == 'a':
            return 0.5
        elif param == 'b':
            return 0.5
        else:
            raise ValueError('Unknown parameter')

class GeometricMean(BinaryOperator):
    r"""Binary operator based on geometric mean.

    The GeometricMean object akes the geometric mean of two :py:class:`Variable`s
    as shown below:

    .. math::

        v = \sqrt{ab}

    This class can also be used to implement mixing rules.

    Parameters
    ----------
    a : :py:class:`Variable`
        The first parameter of the GeometricMean.
    b : :py:class:`Variable`
        The second parameter of the GeometricMean.

    """
    @property
    def value(self):
        return np.sqrt(self.a.value*self.b.value)

    def _derivative(self, param):
        """Calculates the derivative of the specified GeometricMean object with respect to its parameters.

        Parameters
        ----------
        param : str
            The parameter with respect to which to take the derivative.
            (Can only be 'a' or 'b').

        Returns
        -------
        float
            The calculated derivative value.

        Raises
        ------
        ValueError
            If the parameter argument is not 'a' or 'b'.

        """
        if param == 'a':
            try:
                return 0.5*np.sqrt(self.b.value/self.a.value)
            except ZeroDivisionError:
                return np.inf
        elif param == 'b':
            try:
                return 0.5*np.sqrt(self.a.value/self.b.value)
            except ZeroDivisionError:
                return np.inf
        else:
            raise ValueError('Unknown parameter')
