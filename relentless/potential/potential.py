__all__ = ['PairPotential','Tabulator']

import json
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
        self.default = default
        for key in self:
            self[key] = FixedKeyDict(keys=self.params)
            self[key].update(self.default)

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

        """
        data = {}
        for key in self:
            all_params = {}
            for param in self[key]:
                var = self[key][param]
                if isinstance(var, Variable):
                    all_params[param] = {'type':'variable', 'value':var.value,
                                         'const':var.const, 'low':var.low, 'high':var.high}
                elif isinstance(var, (float,int)):
                    all_params[param] = {'type':'scalar', 'value':var}
            data[str(key)] = all_params
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
            If the parameter value type in the data and in self do not match

        """
        with open(filename, 'r') as f:
            data = json.load(f)
        pairs = []
        for key in data:
            pairs.append(eval(key))
        if sorted(self.pairs) != sorted(tuple(pairs)):
            raise KeyError('The type pairs in the data are not exactly the pairs in self')
        for key in data:
            key_tup = eval(key)
            if sorted(self.params) != sorted(data[key].keys()):
                raise KeyError('The parameters in the data are not exactly the parameters in self')
            for param in data[key]:
                var_data = data[key][param]
                var_self = self[key_tup][param]
                if var_self is None:
                    if var_data['type'] == 'variable':
                        self[key_tup][param] = Variable(value=var_data['value'], const=var_data['const'],
                                                        low=var_data['low'], high=var_data['high'])
                    elif var_data['type'] == 'scalar':
                        self[key_tup][param] = var_data['value']
                    else:
                        raise TypeError('The parameter value types in the data and self do not match')
                elif var_data['type'] == 'scalar' and isinstance(var_self, (float,int)):
                    self[key_tup][param] = var_data['value']
                elif var_data['type'] == 'variable' and isinstance(var_self, Variable):
                    self[key_tup][param] = Variable(value=var_data['value'], const=var_data['const'],
                                                    low=var_data['low'], high=var_data['high'])
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
            data = json.load(f)
        t_all = data.keys()
        p = data[list(t_all)[0]].keys()
        for par in t_all:
            par = eval(par)
            if len(par) == len(set(par)):
                t = par
                break
        m = CoefficientMatrix(types=t, params=p)
        m.load(filename)
        return m

    def __setitem__(self, key, value):
        """Set coefficients for the (i,j) pair."""
        for p in value:
            if p not in self.params:
                raise KeyError('Only the known parameters can be set in the coefficient matrix.')
        for p in self.params:
            if p in value:
                self[key][p] = value[p]
            else:
                self[key][p] = self.default[p] if p in self.default else None

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
