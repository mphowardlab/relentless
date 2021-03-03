"""
.. rubric:: Developer notes

To implement a new type of potential, create a class that derives from
:class:`Potential` and implement the required functions. You may also need
to implement a new parameter storage container that derives from :class:`Parameters`
if a suitable one does not already exist.

.. autosummary::
    :nosignatures:

    Potential
    Parameters

.. autoclass:: Potential
    :member-order: bysource
    :members: energy,
        force,
        derivative

.. autoclass:: Parameters
    :members:

"""
import abc
import json

import numpy as np

from relentless import _collections
from relentless import variable

class Parameters:
    """Parameters for types.

    Each type is a :class:`str`. A named list of parameters can be set for type.
    An optional shared value can be set for any of the parameters,
    and this value will be used if the per-type value is not set.

    Parameters
    ----------
    types : tuple[str]
        Types.
    params : tuple[str]
        Required parameters.

    Raises
    ------
    ValueError
        If ``params`` is empty.
    TypeError
        If ``params`` is not only strings.

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
        self._shared = _collections.FixedKeyDict(keys=self.params)

        # per-type params
        self._per_type = _collections.FixedKeyDict(keys=self.types)
        for t in self.types:
            self._per_type[t] = _collections.FixedKeyDict(keys=self.params)

    def evaluate(self, key):
        """Evaluate parameters.

        Parameters
        ----------
        key : str
            Key for which the parameters are evaluated.

        Returns
        -------
        params : dict
            The evaluated parameters.

        Raises
        ------
        TypeError
            If a parameter has a type that can't be evaluated.
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
            if isinstance(v, variable.Variable):
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
        """Save the parameter to a JSON file.

        Parameters
        ----------
        filename : str
            The name of the file to save.

        """
        data = {}
        for key in self:
            data[str(key)] = self.evaluate(key)

        with open(filename, 'w') as f:
            json.dump(data, f, sort_keys=True, indent=4)

    def design_variables(self):
        """Get all unique design variables.

        The unique :class:`~relentless.variable.DesignVariable` variables are
        determined from the parameters of the coefficient matrix and their dependencies.

        Returns
        -------
        tuple[:class:`~relentless.variable.DesignVariable`]
            The unique :class:`~relentless.variable.DesignVariable` variables on which the
            parameters depend.

        """
        d = set()
        for k in self:
            for p in self.params:
                var = self[k][p]
                if isinstance(var, variable.DesignVariable):
                    d.add(var)
                elif isinstance(var, variable.DependentVariable):
                    g = var.dependency_graph()
                    for n in g.nodes:
                        if isinstance(n, variable.DesignVariable):
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
        """:class:`~relentless.FixedKeyDict`: The shared parameters."""
        return self._shared

class Potential(abc.ABC):
    """Abstract base class for interaction potential.

    A :class:`Potential` defines the potential energy abstractly, which can
    be parameterized on a ``key`` (like a type) and that is a function of an
    arbitrary scalar coordinate ``x``. Concrete :meth:`energy`, :meth:`force`,
    :meth:`derivative` methods must be implemented to define the potential
    energy (and its derivatives).

    Parameters
    ----------
    keys : list
        Keys for parameterizing the potential.
    params : list
        Parameters of the potential.
    container : object
        Container for storing coefficients. By default, :class:`Parameters` is used.
        The constructor of the ``container`` must accept two arguments: ``keys``
        and ``params``.

    """
    def __init__(self, keys, params, container=None):
        if container is None:
            container = Parameters
        self.coeff = container(keys,params)

    @abc.abstractmethod
    def energy(self, key, x):
        """Evaluate potential energy.

        Parameters
        ----------
        key
            Key parameterizing the potential in :attr:`coeff<container>`.
        x : float or list
            Potential coordinate.

        Returns
        -------
        float or numpy.ndarray
            The pair energy evaluated at ``x``. The return type is consistent
            with ``x``.

        """
        pass

    @abc.abstractmethod
    def force(self, key, x):
        """Evaluate force magnitude.

        The force is the (negative) magnitude of the ``x`` gradient.

        Parameters
        ----------
        key
            Key parameterizing the potential in :attr:`coeff<container>`.
        x : float or list
            Potential coordinate.

        Returns
        -------
        float or numpy.ndarray
            The force evaluated at ``x``. The return type is consistent
            with ``x``.

        """
        pass

    @abc.abstractmethod
    def derivative(self, key, var, x):
        """Evaluate potential parameter derivative.

        Parameters
        ----------
        key
            Key parameterizing the potential in :attr:`coeff<container>`.
        var : :class:`~relentless.variable.Variable`
            The variable with respect to which the derivative is calculated.
        x : float or list
            Potential energy coordinate.

        Returns
        -------
        float or numpy.ndarray
            The potential parameter derivative evaluated at ``x``. The return
            type is consistent with ``x``.

        """
        pass

    @classmethod
    def _zeros(cls, x):
        """Force input to a 1-dimensional array and make matching array of zeros.

        Parameters
        ----------
        x : float or list
            One-dimensional array of coordinates.

        Returns
        -------
        numpy.ndarray
            ``x`` coerced into a NumPy array.
        numpy.ndarray
            Array of zeros the same shape as ``x``.
        bool
            ``True`` if ``x`` was originally a scalar.

        Raises
        ------
        TypeError
            If x is not a 1-dimensional array

        """
        s = np.isscalar(x)
        x = np.array(x, dtype=np.float64, ndmin=1)
        if len(x.shape) != 1:
            raise TypeError('Potential coordinate must be 1D array.')
        return x,np.zeros_like(x),s

    def save(self, filename):
        """Save the coefficient matrix to file as JSON data.

        Parameters
        ----------
        filename : str
            The name of the file to which to save the data.

        """
        self.coeff.save(filename)
