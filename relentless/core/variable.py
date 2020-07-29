__all__ = ['Variable','DesignVariable','DependentVariable',
           'SameAs',
           'MixingRule','ArithmeticMean','GeometricMean'
          ]

import abc
from enum import Enum
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

        >>> v = Variable(value=1.0, low=0.0)
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
        """State of the variable.

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
        b : :py:class:`Variable.State`
            The state of the variable.

        """
        if self.low is not None and value <= self.low:
            v = self.low
            b = Variable.State.LOW
        elif self.high is not None and value >= self.high:
            v = self.high
            b = Variable.State.HIGH
        else:
            v = value
            b = Variable.State.FREE

        return v,b

    @property
    def value(self):
        """float: The value of the variable"""
        return self._value

    @value.setter
    def value(self, value):
        if not isinstance(value,(float,int)):
            raise ValueError('Variable must be a float or int')
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
        """:py:class:`Variable.State`: The state of the variable"""
        return self._state

    def isfree(self):
        """Confirms if the variable is unconstrained or within the bounds.

        Returns
        -------
        bool
            True if the variable is unconstrained or within the bounds, False otherwise.

        """
        return self.state is Variable.State.FREE

    def atlow(self):
        """Confirms if the variable is at the lower bound.

        Returns
        -------
        bool
            True if the variable is at the lower bound, False otherwise.

        """
        return self.state is Variable.State.LOW

    def athigh(self):
        """Confirms if the variable is at the upper bound.

        Returns
        -------
        bool
            True if the variable is at the upper bound, False otherwise.

        """
        return self.state is Variable.State.HIGH

class DependentVariable(Variable):
    """Abstract base class for a variable that depends on other variables.

    A dependent variable is composed from other variables. In addition to the
    value property, it must also supply a derivative method with respect to
    any of its dependent variables.

    The number of dependencies of the variable are locked at construction, but
    their values can be modified.

    """
    def __init__(self, *vardicts, **vars):
        attrs = {}
        for d in vardicts:
            attrs.update(d)
        attrs.update(**vars)

        for k,v in attrs.items():
            super().__setattr__(k,self._assert_variable(v))
        self._depends = tuple(vars.keys())

    def __setattr__(self, name, value):
        if name != '_depends' and name in self._depends:
            value = self._assert_variable(value)
        super().__setattr__(name,value)

    @property
    def depends(self):
        return tuple([getattr(self,k) for k in self._depends])

    @abc.abstractmethod
    def derivative(self, var):
        pass

    @classmethod
    def _assert_variable(cls, v):
        if not isinstance(v, Variable):
            raise TypeError('Dependent variables can only depend on other variables.')
        return v

class SameAs(DependentVariable):
    def __init__(self, a):
        super().__init__(a=a)

    @property
    def value(self):
        return self.a.value

    def derivative(self, var):
        if var is self.a:
            return 1.0
        else:
            return 0.0

class MixingRule(DependentVariable):
    """Dependent variable based on two parameter values."""
    def __init__(self, a, b):
        super().__init__(a=a, b=b)

class ArithmeticMean(MixingRule):
    """Mixing rule based on arithmetic mean."""
    @property
    def value(self):
        return 0.5*(self.a.value+self.b.value)

    def derivative(self, var):
        if var is self.a:
            return 0.5
        elif var is self.b:
            return 0.5
        else:
            return 0.0

class GeometricMean(MixingRule):
    """Mixing rule based on geometric mean."""
    @property
    def value(self):
        return np.sqrt(self.a.value*self.b.value)

    def derivative(self, obj):
        if obj is self.a:
            return 0.5*np.sqrt(self.b/self.a)
        elif obj is self.b:
            return 0.5*np.sqrt(self.a/self.b)
        else:
            return 0.0
