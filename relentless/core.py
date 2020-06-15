__all__ = ['Interpolator','Variable']

from enum import Enum

import numpy as np
import scipy.interpolate

class Interpolator:
    """Interpolating function.

    Interpolates through a function :math:`y(x)` on the domain
    :math:`a \le x \le b` using Akima splines. Outside this domain, `y` is
    extrapolated as a constant, i.e., :math:`y(x < a) = y(a)` and
    :math:`y(x > b) = y(b)`.

    Parameters
    ----------
    x : array_like
        1-d array of x coordinates that must be continually increasing.
    y : array_like
        1-d array of y coordinates.

    Examples
    --------
    Interpolating the line :math:`y=2x`::

        f = Interpolator(x=(-1,0,1), y=(-2,0,2))

    Evaluating the function::

        >>> f(0.5)
        1.0
        >>> f([-0.5,0.5])
        (-1.0, 1.0)

    Extrapolation::

        >>> f(100)
        2.0

    """
    def __init__(self, x, y):
        self._domain = (x[0],x[-1])
        self._spline = scipy.interpolate.Akima1DInterpolator(x=x, y=y)

    def __call__(self, x):
        """Evaluate the interpolating function.

        Parameters
        ----------
        x : array_like
            1-d array of x coordinates to evaluate.

            If `x` is a scalar, it is promoted to a NumPy array.

        Returns
        -------
        result : float or numpy.ndarray
            Interpolated values having the same form as `x`.

        """
        scalar_x = np.isscalar(x)
        x = np.atleast_1d(x)
        result = np.zeros(len(x))

        # clamp lo
        lo = x < self.domain[0]
        result[lo] = self._spline(self.domain[0])

        # clamp hi
        hi = x > self.domain[1]
        result[hi] = self._spline(self.domain[1])

        # evaluate in between
        flags = np.logical_and(~lo,~hi)
        result[flags] = self._spline(x[flags])

        if scalar_x:
            result = result.item()

        return result

    @property
    def domain(self):
        """tuple The valid domain for interpolation."""
        return self._domain

class PairMatrix:
    """Generic matrix of values per-pair.

    Defines a symmetric matrix of parameters corresponding to `(i,j)` pairs.
    The matrix is essentially a dictionary of dictionaries, keyed on `(i,j)`.
    There is an equivalent virtual entry for `(j,i)`. (The pairs that are
    actually saved have `j >= i`.) The dictionary associated with each pair
    can have any number of entries in it, although a common use case is to have
    the same parameter stored per-pair.

    The pairs in the matrix are frozen from the list of `types` specified when
    the object is constructed. It is an error to access pairs that cannot be
    formed from `types`.

    The pair matrix emulates a dictionary, and its pairs are iterable.

    Parameters
    ----------
    types : array_like
        List of types (typically, a type is a `str`).

    Examples
    --------
    Create a pair matrix::

        m = PairMatrix(types=('A','B'))

    Set a pair matrix value::

        m['A','A']['energy'] = 1.0
        m['A','B']['energy'] = -1.0
        m['B','B']['energy'] = 1.0

    Get a pair matrix value::

        >>> m['A','A']['energy']
        1.0
        >>> m['A','B']['energy']
        -1.0
        >>> m['B','A']['energy']
        -1.0

    Iterate a pair matrix::

        for pair in m:
            m[pair]['mass'] = 1.0

    Multiple parameters are a dictionary::

        >>> m['A','B']
        {'energy': -1.0, 'mass': 1.0}

    Single-type matrix still needs types as a tuple::

        PairMatrix(types=('A',))

    """
    def __init__(self, types):
        self.types = tuple(types)

        # flood data with type pairs
        self._data = {}
        for i in self.types:
            for j in self.types:
                if j >= i:
                    self._data[i,j] = {}

    def _check_key(self, key):
        """Check that a pair key is valid.

        Returns
        -------
        tuple
            The `(i,j)` pair that is keyed in the dictionary.

        Raises
        ------
        KeyError
            If the key is not the right length or is not in the matrix.

        """
        if len(key) != 2:
            raise KeyError('Coefficient matrix requires a pair of types.')

        if key[0] not in self.types:
            raise KeyError('Type {} is not in coefficient matrix.'.format(key[0]))
        elif key[1] not in self.types:
            raise KeyError('Type {} is not in coefficient matrix.'.format(key[1]))

        if key[1] >= key[0]:
            return key
        else:
            return (key[1],key[0])

    def __getitem__(self, key):
        """Get all coefficients for the `(i,j)` pair."""
        i,j = self._check_key(key)
        return self._data[i,j]

    def __setitem__(self, key, value):
        """Set coefficients for the `(i,j)` pair."""
        i,j = self._check_key(key)
        self._data[i,j] = value

    def __iter__(self):
        return iter(self._data)

    def __next__(self):
        return next(self._data)

    def __str__(self):
        return str(self._data)

    @property
    def pairs(self):
        """tuple All unique pairs in the matrix."""
        return tuple(self._data.keys())

class TypeDict:
    """Dictionary with fixed keys set by type.

    Parameters
    ----------
    types : array_like
        List of types (typically, a type is a `str`).
    default
        Initial value to fill in the dictionary, defaults to `None`.

    Examples
    --------
    Create a type dictionary::

        d = TypeDict(types=('A','B'))

    Default values::

        >>> print(d)
        {'A': None, 'B': None}

    Iterate as a dictionary::

        for t in d:
            d[t] = 1.0

    Access by key::

        >>> d['A']
        1.0
        >>> d['B']
        1.0

    Single-type dictionary still needs `types` as a tuple::

        TypeDict(types=('A',))

    """
    def __init__(self, types, default=None):
        self._types = tuple(types)
        self._data = {}
        for i in self.types:
            self._data[i] = default

    def _check_key(self, key):
        """Check that a type is in the dictionary.

        Returns
        -------
        key
            The type that is keyed in the dictionary.

        Raises
        ------
        KeyError
            If the key is not in the dictionary.

        """
        if key not in self.types:
            raise KeyError('Type {} is not in dictionary.'.format(key))
        return key

    def __getitem__(self, key):
        key = self._check_key(key)
        return self._data[key]

    def __setitem__(self, key, value):
        key = self._check_key(key)
        self._data[key] = value

    def __iter__(self):
        return iter(self._data)

    def __next__(self):
        return next(self._data)

    def __str__(self):
        return str(self._data)

    def todict(self):
        """Convert the fixed-key dictionary to a standard dictionary.

        Returns
        -------
        dict
            A copy of the data in the dictionary.

        """
        return dict(self._data)

    @property
    def types(self):
        """tuple All types in the dictionary."""
        return self._types

class Variable:
    """Bounded variable.

    Represents a quantity that optionally takes lower and upper bounds.
    When the value of the quantity is set, these bounds will be respected and
    an internal state will track whether the requested quantity was within or
    outside these bounds. This is useful for performing constrained
    optimization and for ensuring physical quantities have meaningful values
    (e.g., lengths should be positive).

    Parameters
    ----------
    value : float
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
        b : Variable.State
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
        """float The value stored in the variable."""
        return self._value

    @value.setter
    def value(self, value):
        if not isinstance(value,float):
            raise ValueError('Variable must be a float')
        self._value, self._state = self.clamp(float(value))

    @property
    def state(self):
        return self._state

    def isfree(self):
        """True if the variable is within the bounds."""
        return self._state is Variable.State.FREE

    def atlow(self):
        """True if the variable is at the lower bound."""
        return self._state is Variable.State.LOW

    def athigh(self):
        """True if the variable is at the upper bound."""
        return self._state is Variable.State.HIGH
