import abc
import json

import numpy as np

from relentless import core

class Parameters:
    """Parameters for a set of types.

    Defines one or more parameters for a set of types. The parameters can be set
    per-type, or shared between all parameters. The per-type parameters take precedence
    over the shared parameters.

    Parameters
    ----------
    types : array_like
        List of types (A type must be a `str`).
    params : array_like
        List of parameters (A parameter must be a `str`).

    Raises
    ------
    ValueError
        If no types or no parameters are specified.
    TypeError
        If all types and all parameters are not strings.

    """
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
        self._shared = core.FixedKeyDict(keys=self.params)

        # per-type params
        self._per_type = core.FixedKeyDict(keys=self.types)
        for t in self.types:
            self._per_type[t] = core.FixedKeyDict(keys=self.params)

    def evaluate(self, key):
        """Evaluate parameters.

        Returns a dictionary of the parameters and values for a specified key.

        Parameters
        ----------
        key : `str`
            Key for which the parameters are called.

        Returns
        -------
        params : dict
            The evaluated parameters.

        Raises
        ------
        TypeError
            If a parameter is of an unrecognizable type.
        ValueError
            If a parameter is not set for the specified key.

        """
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
            if isinstance(v, core.Variable):
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
                if isinstance(var, core.DesignVariable):
                    d.add(var)
                elif isinstance(var, core.DependentVariable):
                    g = var.dependency_graph()
                    for n in g.nodes:
                        if isinstance(n, core.DesignVariable):
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
    """Abstract base class for interaction potential.

    To implement this class, concrete `energy`, `force`, `derivative` methods must be defined.

    Parameters
    ----------
    types : array_like
        List of types (A type must be a `str`).
    params : array_like
        List of parameters (A parameter must be a `str`).
    container : `callable` class storing types and parameters
        Coefficient matrix class (defaults to `None`, and the container used is :py:class:`Parameters`).

    """
    def __init__(self, types, params, container=None):
        if container is None:
            container = Parameters
        self.coeff = container(types=types, params=params)

    @abc.abstractmethod
    def energy(self, key, x):
        pass

    @abc.abstractmethod
    def force(self, key, x):
        pass

    @abc.abstractmethod
    def derivative(self, key, var, x):
        pass

    @classmethod
    def _zeros(cls, x):
        """Force input to a 1-dimensional array and make matching array of zeros.

        Parameters
        ----------
        x : array_like
            The location(s) provided as input.

        Returns
        -------
        array_like
            The x parameter
        array_like
            An array of 0 with the same shape as x
        bool
            Value indicating if x is scalar

        Raises
        ------
        TypeError
            If x is not a 1-dimensional array

        """
        s = np.isscalar(x)
        x = np.array(x, dtype=np.float64, ndmin=1)
        if len(x.shape) != 1:
            raise TypeError('Expecting 1D array.')
        return x,np.zeros_like(x),s

    def save(self, filename):
        """Saves the coefficient matrix to file as JSON data.

        Parameters
        ----------
        filename : str
            The name of the file to which to save the data.

        """
        self.coeff.save(filename)
