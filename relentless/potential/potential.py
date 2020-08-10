__all__ = ['PairParameters','PairPotential','Tabulator']

import abc
import json
import warnings

import numpy as np

from relentless.core import PairMatrix
from relentless.core import FixedKeyDict
from relentless.core import Variable, DependentVariable, IndependentVariable, DesignVariable

class PairParameters(PairMatrix):
    """Parameters for pairs of types.

    Defines one or more parameters for a set of types. The parameters can be set
    per-pair, per-type, or shared between all pairs. The per-pair parameters take
    precedence over the shared parameters. The per-type parameters are not included in
    the evaluation of pair parameters, but can be used to set per-pair or shared parameters.

    Parameters
    ----------
    types : array_like
        List of types (A type must be a `str`).
    params : array_like
        List of parameters (A parameter must be a `str`).

    Raises
    ------
    ValueError
        If params is initialized as empty
    TypeError
        If params does not consist of only strings

    Examples
    --------
    Create a coefficient matrix with defined types and params::

        m = PairParameters(types=('A','B'), params=('energy','mass'))

    Set coefficient matrix values by accessing parameter directly::

        m['A','A']['energy'] = 2.0

    Assigning to a pair using `update()` overwrites the specified per-pair parameters::

        m['A','A'].update({'mass':2.5})  #does not reset 'energy' value to `None`
        m['A','A'].update(mass=2.5)      #equivalent statement
        >>> print(m['A','A'])
        {'energy':2.0, 'mass':2.5}

    Assigning to a pair using `=` operator overwrites the specified per-pair parameters
    and resets the other parameters::

        m['A','A'] = {'mass':2.5}  #does reset 'energy' value to `None`
        >>> print(m['A','A'])
        {'energy': None, 'mass':2.5}

    Set coefficient matrix values by setting parameters in full::

        m['A','B'] = {'energy':DesignVariable(value=2.0,high=1.5), 'mass':0.5}

    Set coefficient matrix values by iteratively accessing parameters::

        for p in m.params:
            m['B','B'][p] = 0.1

    Evaluate (retrieve) pair parameters::

        >>> print(m.evaluate(('B','B')))
        {'energy':0.1, 'mass':0.1}

    Utilizing `evaluate()` computes the defined values of all objects,
    while directly accessing values returns the objects themselves::

        >>> print(m.evaluate(('A','B')))
        {'energy':2.0, 'mass':0.5}
        >>> print(m['A','B'])
        {'energy':<relentless.core.DesignVariable object at 0x561124456>, 'mass':0.5}

    Assigning to a type sets the specified per-type parameters::

        m['A'].update(energy=1.0, mass=2.0)
        >>> print(m['A'])
        {'energy':1.0, 'mass':2.0}

    Assigning to shared sets the specified shared parameters::

        m.shared['energy'] = 0.5
        >>> print(m.shared)
        {'energy':0.5, 'mass':None}

    Shared parameters will be used in `evaluate()` if the per-pair parameter is not set::

        >>> m['B','B'] = {'mass': 0.1}
        >>> m.shared = {'energy': 0.5}
        >>> print(m['B','B'])
        {'energy': None, 'mass': 0.1}
        >>> print(m.shared)
        {'energy': 0.5, 'mass': None}
        >>> print(m.evaluate(('B','B'))
        {'energy':0.5, 'mass':0.1}

    """
    def __init__(self, types, params):
        super().__init__(types)

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

        # per-pair params
        self._per_pair = PairMatrix(types)
        for key in self:
            self._per_pair[key] = FixedKeyDict(keys=self.params)

    def evaluate(self, pair):
        """Evaluate pair parameters.

        Returns a dictionary of the parameters and values for the specified pair.

        Parameters
        ----------
        pair : tuple
            Pair for which the parameters are called

        Returns
        -------
        params : `dict`
            The parameters evaluated for the specified pair

        Raises
        ------
        TypeError
            If a parameter is of an unrecognizable type
        ValueError
            If a parameter is not set for the specified pair

        Todo
        ----
        1. Allow callable parameters.
        2. Cache output.

        """
        params = {}
        for p in self.params:
            # use pair parameter if set, otherwise use shared parameter
            if self[pair][p] is not None:
                v = self[pair][p]
            elif self.shared[p] is not None:
                v = self.shared[p]
            else:
                raise ValueError('Parameter {} is not set for ({},{}).'.format(p,pair[0],pair[1]))

            # evaluate the variable
            if isinstance(v, Variable):
                params[p] = v.value
            elif np.isscalar(v):
                params[p] = v
            else:
                raise TypeError('Parameter type unrecognized')

            # final check: error if variable is still not set
            if v is None:
                raise ValueError('Parameter {} is not set for ({},{}).'.format(p,pair[0],pair[1]))

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
        """Get parameters for the (i,j) pair."""
        if isinstance(key, str):
            return self._per_type[key]
        else:
            return self._per_pair[key]

    def __setitem__(self, key, value):
        """Set parameters for the (i,j) pair."""
        for p in value:
            if p not in self.params:
                raise KeyError('Only the known parameters can be set in the coefficient matrix.')
        self[key].clear()
        self[key].update(value)

    def __iter__(self):
        return iter(self._per_pair)

    def __next__(self):
        return next(self._per_pair)

    @property
    def shared(self):
        """:py:class:`FixedKeyDict`: The shared parameters."""
        return self._shared

class PairPotential(abc.ABC):
    """Generic pair potential evaluator.

    A PairPotential object is created with coefficients as a PairParameters object.
    This abstract base class can be extended in order to evaluate custom force,
    energy, and derivative/gradient functions.

    Parameters
    ----------
    types : array_like
        List of types (A type must be a `str`).
    params : array_like
        List of parameters (A parameter must be a `str`).

    Todo
    ----
    1. Inspect _energy() call signature for parameters.

    """
    def __init__(self, types, params):
        # force in standard potential parameters if they are not explicitly set
        params = list(params)
        if 'rmin' not in params:
            params.append('rmin')
        if 'rmax' not in params:
            params.append('rmax')
        if 'shift' not in params:
            params.append('shift')

        self.coeff = PairParameters(types, params)
        for p in self.coeff:
            self.coeff[p]['rmin'] = False
            self.coeff[p]['rmax'] = False
            self.coeff[p]['shift'] = False

    def energy(self, pair, r):
        """Evaluate energy for a (i,j) pair.

        If an `rmin` or `rmax` value is set, then any energy evaluated at an `r`
        value greater than `rmax` or less than `rmin` is set to the value of energy
        evaluated or `rmax` or `rmin`, respectively.

        Additionally, if the `shift` parameter is set to be `True`, all energy values
        are shifted so that the energy is `rmax` is 0.

        Parameters
        ----------
        pair : array_like
            The pair for which to calculate the energy.
        r : array_like
            The location(s) at which to evaluate the energy.

        Returns
        -------
        scalar or array_like
            The energy at the specified location(s).
            The returned quantity will be a scalar if `r` is scalar
            or a numpy array if `r` is array_like.

        Raises
        ------
        ValueError
            If any value in `r` is negative.
        ValueError
            If the potential is to be shifted without setting `rmax`.

        """
        params = self.coeff.evaluate(pair)
        r,u,scalar_r = self._zeros(r)
        if any(r < 0):
            raise ValueError('r cannot be negative')

        # evaluate at points below rmax (if set) first, including rmin cutoff (if set)
        flags = np.ones(r.shape[0], dtype=bool)
        if params['rmin'] is not False:
            range_ = r < params['rmin']
            flags[range_] = False
            u[range_] = self._energy(params['rmin'], **params)
        if params['rmax'] is not False:
            flags[r > params['rmax']] = False
        u[flags] = self._energy(r[flags], **params)

        # if rmax is set, truncate or shift depending on the mode
        if params['rmax'] is not False:
            # with shifting, move the whole potential up
            # otherwise, set energy to constant for any r beyond rmax
            if params['shift']:
                u[r <= params['rmax']] -= self._energy(params['rmax'], **params)
            else:
                u[r > params['rmax']] = self._energy(params['rmax'], **params)
        elif params['shift'] is True:
            raise ValueError('Cannot shift potential without rmax')

        # coerce u back into shape of the input
        if scalar_r:
            u = u.item()
        return u

    def force(self, pair, r):
        """Evaluate force for a (i,j) pair.

        The force is only evaluated for `r` values between `rmin` and `rmax`, if set.

        Parameters
        ----------
        pair : array_like
            The pair for which to calculate the force.
        r : array_like
            The location(s) at which to evaluate the force.

        Returns
        -------
        scalar or array_like
            The force at the specified location(s).
            The returned quantity will be a scalar if `r` is scalar
            or a numpy array if `r` is array_like.

        Raises
        ------
        ValueError
            If any value in `r` is negative.

        """
        params = self.coeff.evaluate(pair)
        r,f,scalar_r = self._zeros(r)
        if any(r < 0):
            raise ValueError('r cannot be negative')

        # only evaluate at points inside [rmin,rmax], if specified
        flags = np.ones(r.shape[0], dtype=bool)
        if params['rmin'] is not False:
            flags[r < params['rmin']] = False
        if params['rmax'] is not False:
            flags[r > params['rmax']] = False
        f[flags] = self._force(r[flags], **params)

        # coerce f back into shape of the input
        if scalar_r:
            f = f.item()
        return f

    def derivative(self, pair, var, r):
        """Evaluate derivative for a (i,j) pair with respect to a variable.

        The derivative is only evaluated for `r` values between `rmin` and `rmax`, if set.
        The derivative can only be evaluted with respect to a :py:class:`Variable`.

        Parameters
        ----------
        pair : array_like
            The pair for which to calculate the derivative.
        var : :py:class:`Variable`
            The variable with respect to which the derivative is to be calculated.
        r : array_like
            The location(s) at which to calculate the derivative.

        Returns
        -------
        scalar or array_like
            The derivative at the specified location(s).
            The returned quantity will be a scalar if `r` is scalar
            or a numpy array if `r` is array_like.

        Raises
        ------
        ValueError
            If any value in `r` is negative.

        """
        params = self.coeff.evaluate(pair)
        r,deriv,scalar_r = self._zeros(r)
        if any(r < 0):
            raise ValueError('r cannot be negative')
        if not isinstance(var, Variable):
            raise TypeError('Parameter with respect to which to take the derivative must be a Variable.')

        # only evaluate at points inside [rmin,rmax], if specified
        flags = np.ones(r.shape[0], dtype=bool)
        if params['rmin'] is not False:
            flags[r < params['rmin']] = False
        if params['rmax'] is not False:
            flags[r > params['rmax']] = False

        # evaluate with respect to parameter
        d_params = [i for i in self.coeff.params if not(i=='rmin' or i=='rmax' or i=='shift')]
        for p in self.coeff.params:
            p_obj = self.coeff[pair][p]
            if p=='rmin' or p=='rmax' or p=='shift':
                continue
            elif isinstance(p_obj, DependentVariable):
                deriv[flags] += (self._derivative(p, r[flags], **params)
                                *p_obj.derivative(var))
            elif isinstance(p_obj, IndependentVariable) and var is p_obj:
                deriv[flags] += self._derivative(p, r[flags], **params)

        # coerce derivative back into shape of the input
        if scalar_r:
            deriv = deriv.item()
        return deriv

    def _zeros(self, r):
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

    @abc.abstractmethod
    def _energy(self, r, **params):
        pass

    @abc.abstractmethod
    def _force(self, r, **params):
        pass

    @abc.abstractmethod
    def _derivative(self, param, r, **params):
        pass

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

class Tabulator:
    """Combines and tabulates multiple potentials together.

    Evaluates accumulated energy and force values for multiple potential functions
    at different r values, allows regularization of the force and shifting of the energy.

    Parameters
    ----------
    r : array_like
        The values at which to evaluate energy and force. Must be a 1-D array,
        with values continously increasing.
    fmax : float
        (Optional) The maximum magnitude of the force for regularization at small r values.
    fcut : float
        (Optional) The magnitude of the force for truncation at large r values.
    shift : bool
        If 'True', shift the potential (according to value of fcut) (defaults to `True`).

    Raises
    ------
    ValueError
        If fmax is set, and it is not positive.
    ValueError
        If fcut is set, and it is not positive.

    """
    def __init__(self, r, fmax=None, fcut=None, shift=True):
        if fmax is not None and fmax <= 0:
            raise ValueError('fmax must be positive')
        if fcut is not None and fcut < 0:
            raise ValueError('fcut must be positive')

        self.r = r
        self.fmax = fmax
        self.fcut = fcut
        self.shift = shift

    @property
    def r(self):
        """array_like: The values of r at which to evaluate energy and force."""
        return self._r

    @r.setter
    def r(self, points):
        points = np.array(points)
        if points.ndim > 1:
            raise TypeError('r must be a 1-D array')
        if not np.all(points[1:] > points[:-1]):
            raise ValueError('r values must be continuously increasing')
        self._r = points

    def energy(self, pair, potentials):
        """Evaluates and accumulates energy for all potentials, for the specified pair.

        Parameters
        ----------
        pair : tuple
            The type pair (i,j) for which to calculate the energy.
        potentials : array_like
            All the potential functions for which to calculate and accumulate the energy.

        Returns
        -------
        array_like
            Total energy at each r value.

        """
        u = np.zeros(self.r.shape, dtype=np.float64)
        for pot in potentials:
            try:
                u += pot.energy(pair,self.r)
            except KeyError:
                pass
        return u

    def force(self, pair, potentials):
        """Evaluates and accumulates force for all potentials, for the specified pair.

        Parameters
        ----------
        pair : tuple
            The type pair (i,j) for which to calculate the force.
        potentials : array_like
            All the potential functions for which to calculate and accumulate the force.

        Returns
        -------
        array_like
            Total force at each r value.

        """
        f = np.zeros(self.r.shape, dtype=np.float64)
        for pot in potentials:
            try:
                f += pot.force(pair,self.r)
            except KeyError:
                pass
        return f

    def regularize_force(self, u, f, trim=True):
        """Regularizes and truncates the accumulated energies and forces.

        When shifting is enabled - if fcut is set, then the energies are shifted to be 0 at rcut;
        otherwise, they are shifted to be 0 at rmax.

        Parameters
        ----------
        u : array_like
            Energies, must have the same shape as r.
        f : array_like
            Forces, must have the same shape as r.
        trim : bool
            Whether to trim off trailing zeros from the regularized force/energy values,
            defaults to `True`.

        Returns
        -------
        array_like
            A `nx3` array of columns as r, adjusted u, and regularized f, evaluated in n bins.
        float or `None`
            The value of r at which trimming trailing zeros is implemented or could be
            implemented (if trim is False). Returns the last r value if trimming cannot be performed.

        Raises
        ------
        IndexError
            If the energy array is not the same length as the array of r values.
        IndexError
            If the force array is not the same length as the array of r values.
        UserWarning
            If rmax is too small to cutoff the potential, i.e. if the force at the end
            of the potential is larger than fcut.

        """
        u = np.atleast_1d(u)
        f = np.atleast_1d(f)
        if u.shape != self.r.shape:
            raise IndexError('Potential must have the same length as r.')
        if f.shape != self.r.shape:
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
            if cut < len(f)-1:
                if self.shift:
                    u -= u[cut]
                u[(cut+1):] = 0.
                f[(cut+1):] = 0.
            else:
                warnings.warn('Last tabulated force exceeds fcut, rmax may be too small.', UserWarning)
                if self.shift:
                    u -= u[-1]
        elif self.shift:
            u -= u[-1]

        # trim off trailing zeros
        r = self.r.copy()
        flags = np.abs(np.flip(f)) > 0
        cut = len(f) - np.argmax(flags)
        if cut < len(f):
            rcut = r[cut]
            if trim:
                r = r[:(cut+1)]
                u = u[:(cut+1)]
                f = f[:(cut+1)]
        else:
            rcut = r[-1]

        return np.column_stack((r,u,f)), rcut
