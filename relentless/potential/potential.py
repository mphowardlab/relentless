__all__ = ['Parameters','Potential','PotentialTabulator']

import abc
import json
import warnings

import numpy as np

from relentless.core import PairMatrix
from relentless.core import FixedKeyDict
from relentless.core import Variable, DependentVariable, IndependentVariable, DesignVariable

class Parameters:
    def __init__(self, types, params):
        if len(types) == 0:
            raise ValueError('Cannot initialize with empty types')
        if not all(isinstance(t, str) for t in types):
            raise TypeError('All types must be strings')
        self.types = tuple(types)

        if len(params) == 0:
            raise ValueError('params cannot be initialized as empty')
        if not all(isinstance(p, str) for p in params):
            raise TypeError('All parameters must be strings')
        self.params = tuple(params)

        # shared params
        self._shared = FixedKeyDict(keys=self.params)

        # per-type params
        self._per_type = FixedKeyDict(keys=self.types)
        for t in self.types:
            self._per_type[t] = FixedKeyDict(keys=self.params)

    def evaluate(self, key):
        params = {}
        for p in self.params:
            # use keyed parameter if set, otherwise use shared parameter
            if self[key][p] is not None:
                v = self[key][p]
            elif self.shared[p] is not None:
                v = self.shared[p]
            else:
                raise ValueError('Parameter {} is not set for {}.'.format(p,str(key)))

            # evaluate the variable
            if isinstance(v, Variable):
                params[p] = v.value
            elif np.isscalar(v):
                params[p] = v
            else:
                raise TypeError('Parameter type unrecognized')

            # final check: error if variable is still not set
            if v is None:
                raise ValueError('Parameter {} is not set for {}.'.format(p,str(key)))

        return params

    def save(self, filename):
        """Saves the parameter data to a file.

        Parameters
        ----------
        filename : `str`
            The name of the file to save data in.

        """
        data = {}
        for key in self:
            data[str(key)] = self.evaluate(key)

        with open(filename, 'w') as f:
            json.dump(data, f, sort_keys=True, indent=4)

    def design_variables(self):
        """Get all unique DesignVariables that are the parameters of the coefficient matrix
           or dependendencies of those parameters.

        Returns
        -------
        tuple
            The unique DesignVariables on which the specified PairParameters
            object is dependent.

        """
        d = set()
        for k in self:
            for p in self.params:
                var = self[k][p]
                if isinstance(var, DesignVariable):
                    d.add(var)
                elif isinstance(var, DependentVariable):
                    g = var.dependency_graph()
                    for n in g.nodes:
                        if isinstance(n, DesignVariable):
                            d.add(n)
        return tuple(d)

    def __getitem__(self, key):
        return self._per_type[key]

    def __setitem__(self, key, value):
        for p in value:
            if p not in self.params:
                raise KeyError('Only the known parameters can be set in the coefficient matrix.')
        self[key].clear()
        self[key].update(value)

    def __iter__(self):
        return iter(self._per_type)

    def __next__(self):
        return next(self._per_type)

    @property
    def shared(self):
        """:py:class:`FixedKeyDict`: The shared parameters."""
        return self._shared

class Potential(abc.ABC):
    def __init__(self, types, params, container=None):
        if container is None:
            container = PotentialParameters
        self.coeff = container(types,params)

    @abc.abstractmethod
    def energy(self, key, r):
        pass

    @abc.abstractmethod
    def force(self, key, r):
        pass

    @abc.abstractmethod
    def derivative(self, key, var, r):
        pass

    @classmethod
    def _zeros(cls, r):
        """Force input to a 1-dimensional array and make matching array of zeros.

        Parameters
        ----------
        r : array_like
            The location(s) provided as input.

        Returns
        -------
        array_like
            The r parameter
        array_like
            An array of 0 with the same shape as r
        bool
            Value indicating if r is scalar

        Raises
        ------
        TypeError
            If r is not a 1-dimensional array

        """
        s = np.isscalar(r)
        r = np.array(r, dtype=np.float64, ndmin=1)
        if len(r.shape) != 1:
            raise TypeError('Expecting 1D array for r')
        return r,np.zeros_like(r),s

    def save(self, filename):
        """Saves the coefficient matrix to file as JSON data.

        Parameters
        ----------
        filename : `str`
            The name of the file to which to save the data.

        """
        self.coeff.save(filename)

    def __iter__(self):
        return iter(self.coeff)

    def __next__(self):
        return next(self.coeff)

class PotentialTabulator:
    """Tabulate a potential."""
    def __init__(self, rmax, num_r, potentials=None):
        self.rmax = rmax
        self.num_r = num_r

        if potentials is not None:
            self._potentials = list(potentials)
        else:
            self._potentials = []

    @property
    def potentials(self):
        return self._potentials

    @property
    def rmax(self):
        return self._rmax

    @rmax.setter
    def rmax(self, val):
        if val is not None and val < 0:
            raise ValueError('Maximum radius must be positive.')
        self._rmax = val
        self._compute_r = True

    @property
    def num_r(self):
        return self._num_r

    @num_r.setter
    def num_r(self, val):
        if val is not None and (not isinstance(val,int) or val < 2):
            raise ValueError('Number of points must be at least 2.')
        self._num_r = val
        self._compute_r = True

    @property
    def r(self):
        """array_like: The values of r at which to evaluate energy and force."""
        if self._compute_r:
            if self.rmax is None:
                raise ValueError('Maximum radius must be set.')
            if self.num_r is None:
                raise ValueError('Number of points must be set.')
            self._r = np.linspace(0,self.rmax,self.num_r,dtype=np.float64)
            self._compute_r = False
        return self._r

    def energy(self, key):
        """Evaluates and accumulates energy for all potentials.

        Parameters
        ----------

        Returns
        -------
        array_like
            Total energy at each r value.

        """
        u = np.zeros_like(self.r)
        for pot in self.potentials:
            try:
                u += pot.energy(key,self.r)
            except KeyError:
                pass
        return u

    def force(self, key):
        """Evaluates and accumulates force for all potentials.

        Parameters
        ----------

        Returns
        -------
        array_like
            Total force at each r value.

        """
        f = np.zeros_like(self.r)
        for pot in self.potentials:
            try:
                f += pot.force(key,self.r)
            except KeyError:
                pass
        return f

    def derivative(self, key, var):
        """Evaluates and accumulates derivative for all potentials.

        Parameters
        ----------

        Returns
        -------
        array_like
            Total force at each r value.

        """
        d = np.zeros(self.r)
        for pot in self.potentials:
            try:
                d += pot.derivative(key,var,self.r)
            except KeyError:
                pass
        return d
