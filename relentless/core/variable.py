__all__ = ['Variable','DesignVariable','DependentVariable',
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

class DesignVariable(Variable):
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
        self.const = const
        self.low = low
        self.high = high
        self.value = value

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

    @property
    def value(self):
        """float: The value of the variable"""
        return self._value

    @value.setter
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
        self._depends = {}
        for d in vardicts:
            self._depends.update(d)
        self._depends.update(**kwvars)

        self._attrs = set()
        for k,v in self._depends.items():
            super().__setattr__(k,self._assert_variable(v))
            self._attrs.add(v)

    def __setattr__(self, name, value):
        #Sets the value of the variable on which the specified DependentVariable depends.
        if name != '_depends' and name in self._depends:
            value = self._assert_variable(value)
        super().__setattr__(name,value)

    @property
    def depends(self):
        """dict: The variable or variables on which the specified DependentVariable depends,
                 keyed to the respective parameter names."""
        return self._depends

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
        TypeError
            If the specified DependentVariable object has any circular dependencies.
        ValueError
            The specified DependentVariable does not depend on the variable
            with respect to which to take the derivative.

        """
        #construct graph of DependentVariable dependencies
        chain = nx.DiGraph()
        current_node = self

        #add nodes depth-first
        nodes_visited = set()
        nodes_to_add = set()
        while current_node != None:
            if current_node not in nodes_visited:
                chain.add_node(current_node)
                for neighbor in current_node._attrs:
                    chain.add_node(neighbor)
                nodes_visited.add(current_node)
            #else:
            #    raise TypeError('The specified DependentVariable has circular dependencies.')
            temp_to_add = [i for i in current_node._attrs if isinstance(i, DependentVariable)]
            for j in temp_to_add:
                nodes_to_add.add(j)
            try:
                current_node = nodes_to_add.pop()
            except:
                current_node = None

        #add edges with weights as parameter _derivatives
        for node in chain.nodes:
            if isinstance(node, DependentVariable):
                for k,v in node.depends.items():
                    duplicate_factor = 0
                    for w in node.depends.values():
                        if v is w:
                            duplicate_factor += 1
                    chain.add_edge(node, v, deriv=node._derivative(k)*duplicate_factor)

        #confirm DAG
        if not nx.is_directed_acyclic_graph(chain):
            raise TypeError('The specified DependentVariable has circular dependencies.')

        #compute chain rule
        try:
            paths = nx.all_simple_paths(chain, source=self, target=var)
        except:
            raise ValueError('''The specified DependentVariable does not depend on the
                                variable with respect to which to take the derivative''')
        d = 0.
        for path in map(nx.utils.pairwise, paths):
            temp_d = 1.
            for edge in path:
                temp_d *= chain.edges[edge]['deriv']
            d += temp_d

        return d

    @abc.abstractmethod
    def _derivative(self, var):
        pass

    @classmethod
    def _assert_variable(cls, v):
        #Checks if the dependent variable depends on another variable.
        if not isinstance(v, Variable):
            raise TypeError('Dependent variables can only depend on other variables.')
        return v

class UnaryOperator(DependentVariable):
    """Abstract base class for dependent variable based on one parameter value."""
    def __init__(self, a):
        super().__init__(a=a)

class SameAs(UnaryOperator):
    """Unary operator for copying a variable."""
    @property
    def value(self):
        return self.a.value

    def _derivative(self, var):
        if var == 'a':
            return 1.0
        else:
            return 0.0

class BinaryOperator(DependentVariable):
    """Abstract base class for dependent variable based on two parameter values."""
    def __init__(self, a, b):
        super().__init__(a=a, b=b)

class ArithmeticMean(BinaryOperator):
    """Binary operator based on arithmetic mean."""
    @property
    def value(self):
        return 0.5*(self.a.value+self.b.value)

    def _derivative(self, var):
        if var == 'a':
            return 0.5
        elif var == 'b':
            return 0.5
        else:
            return 0.0

class GeometricMean(BinaryOperator):
    """Binary operator based on geometric mean."""
    @property
    def value(self):
        return np.sqrt(self.a.value*self.b.value)

    def _derivative(self, var):
        if var == 'a':
            return 0.5*np.sqrt(self.b.value/self.a.value)
        elif var == 'b':
            return 0.5*np.sqrt(self.a.value/self.b.value)
        else:
            return 0.0
