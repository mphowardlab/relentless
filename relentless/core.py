__all__ = ['Variable']

import numpy as np
import scipy.interpolate

class Interpolator(object):
    def __init__(self, x, y):
        self._domain = (x[0],x[-1])
        self._spline = scipy.interpolate.Akima1DInterpolator(x=x, y=y)

    def __call__(self, x):
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

        return result

    @property
    def domain(self):
        return self._domain

class PairMatrix(object):
    """ Coefficient matrix.
    """
    def __init__(self, types):
        self.types = tuple(types)

        self._data = {}
        for i in self.types:
            for j in self.types:
                if j >= i:
                    self._data[i,j] = {}

    def _check_key(self, key):
        """ Check that a pair key is valid.
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
        """ Get all coefficients for the (i,j) pair.
        """
        i,j = self._check_key(key)
        return self._data[i,j]

    def __setitem__(self, key, value):
        """ Set coefficients for the (i,j) pair.
        """
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
        return tuple(self._data.keys())

class TypeDict:
    def __init__(self, types, default=None):
        self._types = tuple(types)
        self._data = {}
        for i in self.types:
            self._data[i] = default

    def _check_key(self, key):
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
        return dict(self._data)

    @property
    def types(self):
        return self._types

class Variable(object):
    def __init__(self, name, value=None, low=None, high=None):
        self.name = name
        self.low = low
        self.high = high

        self.value = value

    def check(self, value):
        if self.low is not None and value <= self.low:
            return -1
        elif self.high is not None and value >= self.high:
            return 1
        else:
            return 0

    def clamp(self, value):
        b = self.check(value)
        if b == -1:
            v = self.low
        elif b == 1:
            v = self.high
        else:
            v = value

        return v,b

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if value is not None:
            self._value, self._free = self.clamp(value)
        else:
            self._value = None
            self._free = 0

    def is_free(self):
        return self._free == 0

    def is_low(self):
        return self._free == -1

    def is_high(self):
        return self._free == 1
