from __future__ import division

import warnings

import numdifftools
import numpy as np

class CoefficientMatrix(object):
    """ Pair coefficient matrix.
    """
    def __init__(self, types, params, default={}):
        self.types = tuple(types)
        self.params = tuple(params)

        self._data = {}
        for i in types:
            for j in types:
                self._data[i,j] = {}
                for p in self.params:
                    v = default[p] if p in default else None
                    self._data[i,j][p] = v

    def evaluate(self, pair):
        i,j = self._check_key(pair)

        params = {}
        for p in self.params:
            v = self._data[i,j][p]
            params[p] = v(self) if callable(v) else v
            if params[p] is None:
                raise ValueError('Parameter {} is not set for ({},{}).'.format(p,i,j))

        return params

    def perturb(self, pair, key, param, value):
        # check keys now to bypass later checks
        self._check_key(pair)
        self._check_key(key)

        if param not in self.params:
            raise KeyError('Parameter {} is not part of the coefficient matrix.'.format(param))

        old_value = self._data[key][param]
        if old_value is None:
            raise ValueError('Cannot perturb parameter {}, not set for ({},{}).'.format(param,*key))
        elif callable(old_value):
            raise KeyError('Cannot perturb a callable parameter; it is chained.')

        # evaluate old values first
        old_params = self.evaluate(pair)

        # swap in new value
        self._data[key][param] = value
        new_params = self.evaluate(pair)

        # restore old value
        self._data[key][param] = old_value

        return new_params, old_params

    def copy(self):
        coeff = CoefficientMatrix(types=self.types, params=self.params)
        coeff._data = self._data.copy()
        return coeff

    def _check_key(self, key):
        """ Check that a pair key is valid.
        """
        if len(key) != 2:
            raise KeyError('Coefficient matrix requires a pair of types.')

        if key[0] not in self.types:
            raise KeyError('Type {} is not in coefficient matrix.'.format(key[0]))
        elif key[1] not in self.types:
            raise KeyError('Type {} is not in coefficient matrix.'.format(key[1]))

        return key

    def __getitem__(self, key):
        """ Get all coefficients for the (i,j) pair.
        """
        self._check_key(key)
        return self._data[key]

    def __setitem__(self, key, value):
        """ Set coefficients for the (i,j) pair.
        """
        i,j = self._check_key(key)

        for p in value:
            if p not in self.params:
                raise KeyError('Only the known parameters can be set in coefficient matrix.')
            self._data[i,j][p] = value[p]
            if i != j:
                self._data[j,i][p] = value[p]

    def __iter__(self):
        return iter(self._data)

    def __next__(self):
        return next(self._data)

    def __str__(self):
        return str(self._data)

class PairPotential(object):
    """ Generic pair potential evaluator.
    """
    _id = 0

    def __init__(self, types, params, default={}):
        self.coeff = CoefficientMatrix(types, params, default)
        self.id = PairPotential._id
        PairPotential._id += 1

    def __call__(self, r, pair):
        """ Evaluate energy for a (i,j) pair.
        """
        params = self.coeff.evaluate(pair)
        return self.energy(r, **params)

    def derivative(self, r, pair, key, param):
        """ Evaluate derivative for a (i,j) pair with respect to a key,param parameter.
        """
        if callable(self.coeff[key][param]):
            raise KeyError('Cannot differentiate a callable parameter; it is chained.')

        # setup derivative
        def u(p):
            params,_ = self.coeff.perturb(pair,key,param,p)
            return self.energy(r, **params)
        dudp = numdifftools.Derivative(u)

        # evaluate derivative at current value of param
        # numdifftools raises a deprecation FutureWarning via NumPy, so silence.
        params = self.coeff.evaluate(pair)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            deriv = dudp(params[param])

        return deriv

    def energy(self, r):
        """ Evaluate the potential energy.
        """
        raise NotImplementedError()

    def force(self, r):
        """ Evaluate the force.
        """
        raise NotImplementedError()

    def _zeros(self, r):
        # coerce input shape and create zeros for output
        r = np.asarray(np.atleast_1d(r))
        if len(r.shape) != 1:
            raise TypeError('Expecting 1D array for r')
        return r,np.zeros_like(r)

class NetPotential(PairPotential):
    def __init__(self, potentials=[]):
        self.types = None
        self._potentials = set()

        for pot in potentials:
            self.add(pot)

    def add(self, potential):
        if self.types is None:
            self.types = tuple(sorted(potential.coeff.types))
        else:
            if tuple(sorted(potential.coeff.types)) != self.types:
                raise KeyError('Potentials must all have the same types.')

        self._potentials.add(potential)

    def remove(self, potential):
        self._potentials.remove(potential)

    def __call__(self, r, pair):
        r,u = self._zeros(r)

        # exit early if there are no potentials added
        if len(self._potentials) == 0:
            return u

        # sum up all potentials, noting anything that spills over
        uinf = np.zeros(u.shape, dtype=bool)
        for pot in self._potentials:
            params = pot.coeff.evaluate(pair)
            up = pot(r, **params)

            # todo: worry about neg inf.?
            flags = np.isfinite(up)
            u[flags] += up[flags]
            uinf |= ~flags

        # set to a large (?) value
        u[uinf] = 1000.

        return u

class WCAPotential(PairPotential):
    def __init__(self,types):
        super(WCAPotential,self).__init__(types=types,
                                          params=('epsilon','sigma','n'),
                                          default={'n': 6})

    def energy(self, r, epsilon, sigma, n):
        r,u = self._zeros(r)

        # evaluate cutoff potential
        rcut = 2.**(1./n)*sigma
        flags = r <= rcut
        rn_inv = np.power(sigma/r[flags], n)
        u[flags] = 4.*epsilon*(rn_inv**2 - rn_inv + 0.25)

        return u

    def force(self, r, epsilon, sigma, n):
        r,f = self._zeros(r)

        # evaluate cutoff force
        rcut = 2.**(1./n)*sigma
        flags = r <= rcut
        rinv = 1./r[flags]
        rn_inv = np.power(sigma*rinv, n)
        f[flags] = (8.*n*epsilon*rinv)*(rn_inv**2-0.5*rn_inv)

        return f
