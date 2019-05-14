import numpy as np
import scipy.interpolate

class Interpolator(object):
    def __init__(self, gr):
        r = gr[:,0]
        g = gr[:,1]
        self.rmin = r[0]
        self.rmax = r[-1]
        self._spline = scipy.interpolate.Akima1DInterpolator(x=r, y=g)

    def __call__(self, r):
        return self._spline(r)

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
