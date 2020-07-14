__all__ = ['PairPotential','Tabulator']

import abc
import json
import warnings

import numpy as np

from relentless.core import PairMatrix
from relentless.core import FixedKeyDict
from relentless.core import Variable

class CoefficientMatrix(PairMatrix):
    """ Pair coefficient matrix.

    Defines a matrix of PairMatrix objects with one or more parameters.
    Any or all of the parameters can be initialized with a default value.

    Parameters
    ----------
    types : array_like
        List of types (A type must be a `str`).
    params : array_like
        List of parameters (A parameter must be a `str`).
    default : dict
        Initial value for any or all parameters, the values default to `None`.

    Raises
    ------
    ValueError
        If params is initialized as empty
    TypeError
        If params does not consist of only strings

    Examples
    --------
    Create a coefficient matrix with defined types and params::

        m = CoefficientMatrix(types=('A','B'), params=('energy','mass'))

    Create a coefficient matrix with default parameter values::

        m = CoefficientMatrix(types=('A','B'), params=('energy','mass'),
                              default={'energy':0.0, 'mass':0.0})

    Set coefficient matrix values by accessing parameter directly::

        m['A','A']['energy'] = 2.0

    Assigning to a pair using `update()` overwrites the specified parameters::

        m['A','A'].update({'mass':2.5})  #does not reset 'energy' value to default
        m['A','A'].update(mass=2.5)      #equivalent statement
        >>> print(m['A','A'])
        {'energy':2.0, 'mass':2.5}

    Assigning to a pair using `=` operator overwrites the specified parameters and resets
    the other parameters to their defaults, if it exists::

        m['A','A'] = {'mass':2.5}  #does reset 'energy' value to default
        >>> print(m['A','A'])
        {'energy':0.0, 'mass':2.5}

    Set coefficient matrix values by setting parameters in full::

        m['A','B'] = {'energy':Variable(value=2.0,high=1.5), 'mass':0.5}

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
        {'energy':<relentless.core.Variable object at 0x561124456>, 'mass':0.5}

    """
    def __init__(self, types, params, default={}):
        super().__init__(types)

        if len(params) == 0:
            raise ValueError('params cannot be initialized as empty')
        if not all(isinstance(p, str) for p in params):
            raise TypeError('All parameters must be strings')

        self.params = tuple(params)
        self.default = FixedKeyDict(keys=self.params)
        self.default.update(default)

        for key in self:
            self._data[key] = FixedKeyDict(keys=self.params)
            self._data[key].update(self.default)

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
        """Saves the data from a coefficient matrix to a file.

        Serializes each parameter value in order to track the internal values
        of the :py:class:`Variable` object and distinguish it from 'scalar' values.

        Parameters
        ----------
        filename : `str`
            The name of the file to save data in.

        Raises
        ------
        TypeError
            If a value is of an unrecognizable type

        """
        data = {}
        for key in self:
            dkey = str(key)
            data[dkey] = {}
            for param in self[key]:
                var = self[key][param]
                if isinstance(var, Variable):
                    data[dkey][param] = {'type':'variable', 'value':var.value,
                                         'const':var.const, 'low':var.low, 'high':var.high}
                elif np.isscalar(var):
                    data[dkey][param] = {'type':'scalar', 'value':var}
                else:
                    raise TypeError('Value type unrecognized')

        with open(filename, 'w') as f:
            json.dump(data, f, sort_keys=True, indent=4)

    def load(self, filename):
        """Loads the data from a file into the coefficient matrix.

        Parameters
        ----------
        filename : `str`
            The name of the file from which to load data.

        Raises
        ------
        KeyError
            If the pairs in the data are not exactly the pairs in self
        KeyError
            If the params in the data are not exactly the params in self
        TypeError
            If the value type in the data is of an unrecognized type
        TypeError
            If the parameter value type in the data and in self do not match

        """
        with open(filename, 'r') as f:
            data_ = json.load(f)
        data = {eval(key): data_[key] for key in data_}

        if set(self.pairs) != set(data.keys()):
            raise KeyError('The type pairs in the data are not exactly the pairs in self')
        for key in data:
            if set(self.params) != set(data[key].keys()):
                raise KeyError('The parameters in the data are not exactly the parameters in self')

            for param in data[key]:
                var_data = data[key][param]
                if self[key][param] is None:
                    if var_data['type'] == 'variable':
                        self[key][param] = Variable(value=var_data['value'], const=var_data['const'],
                                                    low=var_data['low'], high=var_data['high'])
                    elif var_data['type'] == 'scalar':
                        self[key][param] = var_data['value']
                    else:
                        raise TypeError('Unrecognized value type in the data')
                elif var_data['type'] == 'variable' and isinstance(self[key][param], Variable):
                    self[key][param].value = var_data['value']
                    self[key][param].const = var_data['const']
                    self[key][param].low = var_data['low']
                    self[key][param].high = var_data['high']
                elif var_data['type'] == 'scalar' and np.isscalar(self[key][param]):
                    self[key][param] = var_data['value']
                else:
                    raise TypeError('The parameter value types in the data and self do not match')

    @classmethod
    def from_file(cls, filename):
        """Load data from a file into a new coefficient matrix object.

        Parameters
        ----------
        filename : `str`
            The name of the file from which to load data.

        Returns
        -------
        :py:class:`CoefficientMatrix`
            Object initialized with the data in the file.

        """
        with open(filename, 'r') as f:
            data_ = json.load(f)
        data = {eval(key): data_[key] for key in data_}

        #get the types and parameters from the JSON file
        par = None
        typ = set()
        for key in data:
            for t in key:
                typ.add(t)
            if par is None:
                par = data[key].keys()
        typ = tuple(typ)

        m = CoefficientMatrix(types=typ, params=par)
        m.load(filename)
        return m

    def __setitem__(self, key, value):
        """Set coefficients for the (i,j) pair."""
        for p in value:
            if p not in self.params:
                raise KeyError('Only the known parameters can be set in the coefficient matrix.')

        self[key].update(self.default)
        self[key].update(value)

class PairPotential(abc.ABC):
    """Generic pair potential evaluator.

    A PairPotential object is created with coefficients as a CoefficientMatrix.
    This abstract base class can be extended in order to evaluate custom force, energy,
    and derivative/gradient functions.

    Parameters
    ----------
    types : array_like
        List of types (A type must be a `str`).
    params : array_like
        List of parameters (A parameter must be a `str`).
    default : dict
        Initial value for any or all parameters. Any value not specified defaults to `None`.

    Todo
    ----
    1. Inspect _energy() call signature for parameters.

    """
    def __init__(self, types, params, default={}):
        # force in standard potential parameters if they are not explicitly set
        params = list(params)
        # rmin defaults to False
        if 'rmin' not in params:
            params.append('rmin')
        if 'rmin' not in default:
            default['rmin'] = False
        # rmax defaults to False
        if 'rmax' not in params:
            params.append('rmax')
        if 'rmax' not in default:
            default['rmax'] = False
        # shift defaults to False
        if 'shift' not in params:
            params.append('shift')
        if 'shift' not in default:
            default['shift'] = False

        self.coeff = CoefficientMatrix(types, params, default)

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

    def derivative(self, pair, param, r):
        """Evaluate derivative for a (i,j) pair with respect to a parameter.

        The derivative is only evaluated for `r` values between `rmin` and `rmax`, if set.

        Parameters
        ----------
        pair : array_like
            The pair for which to calculate the derivative.
        param : `str`
            The parameter with respect to which the derivative is to be calculated.
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

        # only evaluate at points inside [rmin,rmax], if specified
        flags = np.ones(r.shape[0], dtype=bool)
        if params['rmin'] is not False:
            flags[r < params['rmin']] = False
        if params['rmax'] is not False:
            flags[r > params['rmax']] = False
        deriv[flags] = self._derivative(param, r[flags], **params)

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
        r = np.atleast_1d(r)
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

    def load(self, filename):
        """Loads the coefficient matrix from JSON data in file.

        Parameters
        ----------
        filename : `str`
            The name of the file from which to load data.

        """
        self.coeff.load(filename)

    def __iter__(self):
        return iter(self.coeff)

    def __next__(self):
        return next(self.coeff)

class Tabulator:
    """Combines and tabulates multiple potentials together.

    Evaluates accumulated energy and force values for multiple potential functions at different r values,
    allows regularization of the force and shifting of the energy.

    Parameters
    ----------
    nbins : int
        The number of bins in which to place the values for the potential.
    rmin : float
        The minimum r value at which to calculate energy and force.
    rmax : float
        The maximum r value at which to calculate energy and force.
    fmax : float
        (Optional) The maximum magnitude of the force for regularization at small r values.
    fcut : float
        (Optional) The magnitude of the force for truncation at large r values.
    edges : bool
        If `True`, evaluate r at edges of the bins; otherwise, evaluate at the centers (defaults to `True`).
    shift : bool
        If 'True', shift the potential (according to value of fcut) (defaults to `True`).

    Raises
    ------
    ValueError
        If nbins is not a positive integer.
    ValueError
        If rmin is greater than or equal to rmax.
    ValueError
        If rmin is not positive.
    ValueError
        If fmax is set, and it is not positive.
    ValueError
        If fcut is set, and it is not positive.

    """
    def __init__(self, nbins, rmin, rmax, fmax=None, fcut=None, edges=True, shift=True):
        if not isinstance(nbins, int) and nbins > 0:
            raise ValueError('nbins must be a positive integer')
        if rmin >= rmax:
            raise ValueError('rmin must be less than rmax')
        if rmin < 0:
            raise ValueError('rmin must be positive')
        if (fmax is not None) and (fmax <= 0):
            raise ValueError('fmax must be positive')
        if (fcut is not None) and (fcut <= 0):
            raise ValueError('fcut must be positive')

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

        self._shift = shift

    @property
    def dr(self):
        """float: The interval for r between each bin."""
        return self._dr

    @property
    def r(self):
        """array_like: The r values at each bin."""
        return self._r

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
        u = np.zeros_like(self.r)
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
        f = np.zeros_like(self.r)
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
        float or `None`
            The value of r at which trimming trailing zeros is implemented or could be
            implemented (if trim is False). Returns the last r value if trimming cannot be performed.
        array_like
            Stacks the r, adjusted u, and adjusted f as columns into a `nx3` array for n bins.

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
            flags = np.abs(np.flip(f)) <= self.fcut
            cut = len(f) - np.argmax(flags)
            if cut < len(f):
                if self._shift:
                    u -= u[cut]
                u[cut:] = 0.
                f[cut:] = 0.
            else:
                warnings.warn('Last tabulated force exceeds fcut, rmax may be too small.', UserWarning)
                if self._shift:
                    u -= u[-1]
        elif self._shift:
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
