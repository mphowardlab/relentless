__all__ = ['PairPotential','Tabulator']

import json

import numpy as np

from relentless.core import PairMatrix
from relentless.core import Variable

class CoefficientMatrix(PairMatrix):
    """ Pair coefficient matrix."""
    def __init__(self, types, params, default={}):
        super().__init__(types)
        self.params = tuple(params)
        for key in self:
            for p in self.params:
                v = default[p] if p in default else None
                self[key][p] = v

    def evaluate(self, pair):
        """Evaluate pair parameters.

        Todo
        ----
        1. Allow callable parameters.
        2. Cache output.
        """
        params = {}
        for p in self.params:
            v = self[pair][p]
            if isinstance(v, Variable):
                params[p] = v.value
            elif np.isscalar(v):
                params[p] = v
            elif v is not None:
                raise TypeError('Parameter type unrecognized')

            if v is None:
                raise ValueError('Parameter {} is not set for ({},{}).'.format(p,pair[0],pair[1]))

        return params

    def save(self, filename):
        all_params = {}
        for key in self:
            all_params[str(key)] = self.evaluate(key)
        with open(filename, 'w') as f:
            json.dump(all_params, f, sort_keys=True, indent=4)

    def load(self, filename):
        raise NotImplementedError('Load is not implemented')

    def __setitem__(self, key, value):
        """ Set coefficients for the (i,j) pair."""
        for p in value:
            if p not in self.params:
                raise KeyError('Only the known parameters can be set in coefficient matrix.')
            self[key][p] = value[p]

class PairPotential:
    """Generic pair potential evaluator.

    Todo
    ----
    1. Inspect _energy() call signature for parameters.
    """
    _id = 0

    def __init__(self, types, params, default={}):
        # force in standard potential parameters if they are not explicitly set
        # rmin defaults to None
        if 'rmin' not in params:
            params.append('rmin')
        if 'rmin' not in default:
            default['rmin'] = None
        # rmax defaults to None
        if 'rmax' not in params:
            params.append('rmax')
        if 'rmax' not in default:
            default['rmax'] = None
        # shift defaults to False
        if 'shift' not in params:
            params.append('shift')
        if 'shift' not in default:
            default['shift'] = False

        self.coeff = CoefficientMatrix(types, params, default)

        self.id = PairPotential._id
        PairPotential._id += 1

    def energy(self, pair, r):
        """ Evaluate energy for a (i,j) pair."""
        params = self.coeff.evaluate(pair)
        r,u,scalar_r = self._zeros(r)

        # evaluate at points below rmax (if set) first, including rmin cutoff (if set)
        flags = np.ones(r.shape[0], dtype=bool)
        if params['rmin'] is not None:
            range_ = r < params['rmin']
            flags[range_] = False
            u[range_] = self._energy(params['rmin'], **params)
        if params['rmax'] is not None:
            flags[r > params['rmax']] = False
        u[flags] = self._energy(r[flags], **params)

        # if rmax is set, truncate or shift depending on the mode
        if params['rmax'] is not None:
            # with shifting, move the whole potential up
            # otherwise, set energy to constant for any r beyond rmax
            if self.shift:
                u[r <= params['rmax']] -= self._energy(params['rmax'], **params)
            else:
                u[r > params['rmax']] = self._energy(params['rmax'], **params)

        # coerce u back into shape of the input
        if scalar_r:
            u = u.item()
        return u

    def force(self, pair, r):
        """ Evaluate the force for a (i,j) pair."""
        params = self.coeff.evaluate(pair)
        r,f,scalar_r = self._zeros(r)

        # only evaluate at points inside [rmin,rmax], if specified
        flags = np.ones(r.shape[0], dtype=bool)
        if params['rmin'] is not None:
            flags[r < params['rmin']] = False
        if params['rmax'] is not None:
            flags[r > params['rmax']] = False
        f[flags] = self._force(r[flags], **params)

        # coerce f back into shape of the input
        if scalar_r:
            f = f.item()
        return f

    def derivative(self, pair, param, r):
        """ Evaluate derivative for a (i,j) pair with respect to a parameter."""
        params = self.coeff.evaluate(pair)
        r,deriv,scalar_r = self._zeros(r)

        # only evaluate at points inside [rmin,rmax], if specified
        flags = np.ones(r.shape[0], dtype=bool)
        if params['rmin'] is not None:
            flags[r < params['rmin']] = False
        if params['rmax'] is not None:
            flags[r > params['rmax']] = False
        deriv[flags] = self._derivative(param, r[flags], **params)

        # coerce derivative back into shape of the input
        if scalar_r:
            deriv = deriv.item()
        return deriv

    def _zeros(self, r):
        # coerce input shape and create zeros for output
        s = np.isscalar(r)
        r = np.atleast_1d(r)
        if len(r.shape) != 1:
            raise TypeError('Expecting 1D array for r')
        return r,np.zeros_like(r),s

    def _energy(self, r, **params):
        raise NotImplementedError()

    def _force(self, r, **params):
        raise NotImplementedError()

    def _derivative(self, param, r, **params):
        raise NotImplementedError()

    def save(self):
        self.coeff.save('{}.{}.json'.format(self.id,self.__class__.__name__))

    def load(self):
        self.coeff.load('{}.{}.json'.format(self.id,self.__class__.__name__))

    def __iter__(self):
        return iter(self.coeff)

    def __next__(self):
        return next(self.coeff)

class Tabulator:
    def __init__(self, nbins, rmin, rmax, fmax=None, fcut=None, edges=True):
        self._nbins = nbins
        self._rmin = rmin
        self._rmax = rmax

        self.fmax = fmax
        self.fcut = fcut

        self._dr = (rmax-rmin)/nbins
        if edges:
            self._r = np.linspace(rmin, rmax, nbins+1, dtype=np.float64)
        else:
            self._r = rmin + self._dr*(np.arange(nbins, dtype=np.float64)+0.5)

    @property
    def dr(self):
        return self._dr

    @property
    def r(self):
        return self._r

    def energy(self, pair, potentials):
        u = np.zeros_like(self.r)
        for pot in potentials:
            try:
                u += pot.energy(pair,self.r)
            except KeyError:
                pass
        return u

    def force(self, pair, potentials):
        f = np.zeros_like(self.r)
        for pot in potentials:
            try:
                f += pot.force(pair,self.r)
            except KeyError:
                pass
        return f

    def regularize(self, u, f, trim=True):
        if len(u) != len(self.r):
            raise IndexError('Potential must have the same length as r.')
        if len(f) != len(self.r):
            raise IndexError('Force must have the same length as r.')

        # find first point from beginning that is within energy tolerance
        if self.fmax is not None:
            cut = np.argmax(np.abs(f) <= self.fmax)
            if cut > 0:
                u[:cut] = u[cut] - f[cut]*(self.r[:cut] - self.r[cut])
                f[:cut] = f[cut]

        # find first point from end with sufficient force and cutoff the potential after it
        if self.fcut is not None:
            flags = np.abs(np.flip(f)) >= self.fcut
            cut = len(f)-1 - np.argmax(flags)
            u -= u[cut]
            if cut < len(f)-1:
                u[(cut+1):] = 0.
                f[(cut+1):] = 0.

        # trim off trailing zeros
        r = self.r.copy()
        if trim:
            flags = np.abs(np.flip(f)) > 0
            cut = len(f) - np.argmax(flags)
            if cut < len(f)-1:
                r = r[:(cut+1)]
                u = u[:(cut+1)]
                f = f[:(cut+1)]

        return np.column_stack((r,u,f))