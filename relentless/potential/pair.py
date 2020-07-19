import abc

import numpy as np

from relentless import _collections
from relentless import _math
from relentless import variable
from . import potential

class PairParameters(potential.Parameters):
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
        {'energy':<relentless.variable.DesignVariable object at 0x561124456>, 'mass':0.5}

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
        super().__init__(types, params)

        # per-pair params
        self._per_pair = _collections.PairMatrix(types)
        for pair in self:
            self._per_pair[pair] = _collections.FixedKeyDict(keys=self.params)

    def __getitem__(self, pair):
        """Get parameters for the (i,j) pair."""
        if isinstance(pair, str):
            return self._per_type[pair]
        else:
            return self._per_pair[pair]

    def __iter__(self):
        return iter(self._per_pair)

    def __next__(self):
        return next(self._per_pair)

    @property
    def pairs(self):
        return self._per_pair.pairs

class PairPotential(potential.Potential):
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

        super().__init__(types, params, PairParameters)

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
            If the potential is shifted without setting `rmax`.

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
        TypeError
            If the parameter with respect to which to take the derivative
            is not a :py:class:`Variable`.
        ValueError
            If the potential is shifted without setting `rmax`.

        """
        params = self.coeff.evaluate(pair)
        r,deriv,scalar_r = self._zeros(r)
        if any(r < 0):
            raise ValueError('r cannot be negative')
        if not isinstance(var, variable.Variable):
            raise TypeError('Parameter with respect to which to take the derivative must be a Variable.')

        flags = np.ones(r.shape[0], dtype=bool)

        for p in self.coeff.params:
            # skip shift parameter
            if p == 'shift':
                continue

            # try to take chain rule w.r.t. variable first
            p_obj = self.coeff[pair][p]
            if isinstance(p_obj, variable.DependentVariable):
                dp_dvar = p_obj.derivative(var)
            elif isinstance(p_obj, variable.IndependentVariable) and var is p_obj:
                dp_dvar = 1.0
            else:
                dp_dvar = 0.0

            # skip when dp_dvar is exactly zero, since this does not contribute
            if dp_dvar == 0.0:
                continue

            # now take the parameter derivative
            if p=='rmin':
                # rmin deriv
                flags = r < params['rmin']
                deriv[flags] += -self._force(params['rmin'], **params)*dp_dvar
            if p=='rmax':
                # rmax deriv
                if params['shift']:
                    flags = r <= params['rmax']
                    deriv[flags] += self._force(params['rmax'], **params)*dp_dvar
                else:
                    flags = r > params['rmax']
                    deriv[flags] += -self._force(params['rmax'], **params)*dp_dvar
            else:
                # regular parameter derivative
                below = np.zeros(r.shape[0], dtype=bool)
                if params['rmin'] is not False:
                    below = r < params['rmin']
                    deriv[below] += self._derivative(p, params['rmin'], **params)*dp_dvar
                above = np.zeros(r.shape[0], dtype=bool)
                if params['rmax'] is not False:
                    above = r > params['rmax']
                    deriv[above] += self._derivative(p, params['rmax'], **params)*dp_dvar
                elif params['shift']:
                    raise ValueError('Cannot shift without setting rmax.')
                flags = np.logical_and(~below, ~above)
                deriv[flags] += self._derivative(p, r[flags], **params)*dp_dvar
                if params['shift']:
                    deriv -= self._derivative(p, params['rmax'], **params)*dp_dvar

        # coerce derivative back into shape of the input
        if scalar_r:
            deriv = deriv.item()
        return deriv

    @abc.abstractmethod
    def _energy(self, r, **params):
        pass

    @abc.abstractmethod
    def _force(self, r, **params):
        pass

    @abc.abstractmethod
    def _derivative(self, param, r, **params):
        pass

class LennardJones(PairPotential):
    r"""Lennard-Jones 12-6 pair potential.

    .. math::

        u(r) = 4 \varepsilon\left[\left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^6 \right]

    The required coefficients per pair are:

    - :math:`\varepsilon` (``epsilon``): interaction energy
    - :math:`\sigma` (``sigma``): interaction length scale (e.g., particle diameter)

    The optional coefficients per pair are:

    - :math:`r_{\rm min}` (``rmin``): minimum radius, energy and force are 0 for
      :math:`r < r_{\rm min}`. Ignored if ``False`` (default).
    - :math:`r_{\rm max}` (``rmax``): maximum radius, energy and force are 0 for
      :math:`r > r_{\rm max}`. Ignored if ``False`` (default).
    - ``shift``: If ``True``, shift potential to zero at ``rmax`` (default is ``False``).

    Setting :math:`r_{\rm max} = 2^{1/6}\sigma` and ``shift = True`` will give the purely repulsive
    Weeks-Chandler-Anderson potential, which can model nearly hard spheres.

    Parameters
    ----------
    types : array_like
        List of types (A type must be a `str`).

    """
    def __init__(self, types):
        super().__init__(types=types, params=('epsilon','sigma'))

    def _energy(self, r, epsilon, sigma, **params):
        """Evaluates the Lennard-Jones potential energy.

        Parameters
        ----------
        r : array_like
            The value or values of r at which to evaluate the energy.
        epsilon : float or int
            The epsilon parameter for the potential function.
        sigma : float or int
            The sigma parameter for the potential function.

        Returns
        -------
        array_like
            Returns the value or values of the energy evaluated at r.

        Raises
        ------
        ValueError
            If sigma is not a positive number.

        """
        if sigma < 0:
            raise ValueError('sigma must be positive')
        r,u,s = self._zeros(r)
        flags = ~np.isclose(r,0)

        # evaluate potential
        r6_inv = np.power(sigma/r[flags], 6)
        u[flags] = 4.*epsilon*(r6_inv**2 - r6_inv)
        u[~flags] = np.inf

        if s:
            u = u.item()
        return u

    def _force(self, r, epsilon, sigma, **params):
        """Evaluates the Lennard-Jones force.

        Parameters
        ----------
        r : array_like
            The value or values of r at which to evaluate the force.
        epsilon : float or int
            The epsilon parameter for the potential function.
        sigma : float or int
            The sigma parameter for the potential function.

        Returns
        -------
        array_like
            Returns the value or values of the force evaluated at r.

        Raises
        ------
        ValueError
            If sigma is not a positive number.

        """
        if sigma < 0:
            raise ValueError('sigma must be positive')
        r,f,s = self._zeros(r)
        flags = ~np.isclose(r,0)

        # evaluate force
        rinv = 1./r[flags]
        r6_inv = np.power(sigma*rinv, 6)
        f[flags] = (48.*epsilon*rinv)*(r6_inv**2-0.5*r6_inv)
        f[~flags] = np.inf

        if s:
            f = f.item()
        return f

    def _derivative(self, param, r, epsilon, sigma, **params):
        """Evaluates the Lennard-Jones parameter derivative.

        Parameters
        ----------
        param : `str`
            The parameter with respect to which to take the derivative.
        r : array_like
            The value or values of r at which to evaluate the derivative.
        epsilon : float or int
            The epsilon parameter for the potential function.
        sigma : float or int
            The sigma parameter for the potential function.

        Returns
        -------
        array_like
            Returns the value or values of the derivative evaluated at r.

        Raises
        ------
        ValueError
            If sigma is not a positive number.
        ValueError
            If the parameter with respect to which to take the derivative is not 'sigma' or 'epsilon'.

        """
        if sigma < 0:
            raise ValueError('sigma must be positive')
        r,d,s = self._zeros(r)
        flags = ~np.isclose(r,0)

        # evaluate derivative
        r6_inv = np.power(sigma/r[flags], 6)
        if param == 'epsilon':
            d[flags] = 4*(r6_inv**2 - r6_inv)
        elif param == 'sigma':
            d[flags] = (48.*epsilon/sigma)*(r6_inv**2 - 0.5*r6_inv)
        else:
            raise ValueError('The Lennard-Jones parameters are sigma and epsilon.')
        d[~flags] = np.inf

        if s:
            d = d.item()
        return d

class Spline(PairPotential):
    """Spline potential using Akima interpolation.

    Parameters
    ----------
    types : array_like
        List of types (A type must be a str).
    num_knots : int
        The number of knots with which to interpolate.
    mode : str
        Describes the type of design parameter used in the optimization.
        Setting the mode to 'value' uses the knot amplitudes, while 'diff' uses
        the differences between neighboring knot amplitudes. (Defaults to 'diff').

    Raises
    ------
    ValueError
        If the number of knots is not an integer >= 2.
    ValueError
        If the mode is not set to either 'value' or 'diff'.

    """
    valid_modes = ('value','diff')

    def __init__(self, types, num_knots, mode='diff'):
        if isinstance(num_knots,int) and num_knots >= 2:
            self._num_knots = num_knots
        else:
            raise ValueError('Number of spline knots must be an integer >= 2.')

        if mode in self.valid_modes:
            self._mode = mode
        else:
            raise ValueError('Invalid parameter mode, choose from: ' + ','.join(self.valid_modes))

        params = []
        for i in range(self.num_knots):
            ri,ki = self._knot_params(i)
            params.append(ri)
            params.append(ki)
        super().__init__(types=types,params=params)

    def from_array(self, pair, r, u):
        """Sets up the potential from arrays of knot positions and energies.

        Parameters
        ----------
        pair : tuple
            The type pair (i,j) for which to set up the potential.
        r : array_like
            The array of knot positions.
        u : array_like
            The array of energies at each knot position.

        Raises
        ------
        ValueError
            If the array of r values does not have the same length as the number of knots.
        ValueError
            If the array of u values does not have the same length as the number of knots.

        """
        # check that r and u have the right shape
        if len(r) != self.num_knots:
            raise ValueError('r must have the same length as the number of knots')
        if len(u) != self.num_knots:
            raise ValueError('u must have the same length as the number of knots')

        # convert to r,knot form given the mode
        rs = np.asarray(r, dtype=np.float64)
        ks = np.asarray(u, dtype=np.float64)
        if self.mode == 'diff':
            # difference is next knot minus my knot, with last knot fixed at its current value
            ks[:-1] -= ks[1:]

        # make all knots variables, but hold all r and the last knot const
        for i in range(self.num_knots):
            ri,ki = self._knot_params(i)
            if self.coeff[pair][ri] is None:
                self.coeff[pair][ri] = variable.DesignVariable(rs[i],const=True)
            else:
                self.coeff[pair][ri].value = rs[i]
            if self.coeff[pair][ki] is None:
                self.coeff[pair][ki] = variable.DesignVariable(ks[i],const=(i==self.num_knots-1))
            else:
                self.coeff[pair][ki].value = ks[i]

    def _knot_params(self, i):
        """Get the parameter names for a given knot.

        Parameters
        ----------
        i : int
            Key for the knot variable.

        Returns
        -------
        str
            The keyed r value.
        str
            The keyed knot value.

        Raises
        ------
        TypeError
            If the knot key is not an integer.

        """
        if not isinstance(i, int):
            raise TypeError('Knots are keyed by integers')
        return 'r-{}'.format(i), 'knot-{}'.format(i)

    def _energy(self, r, **params):
        """Evaluates the spline potential energy.

        Parameters
        ----------
        r : array_like
            The value or values of r at which to evaluate the energy.
        params : dict
            The parameters (r and knot values) for the potential function.

        Returns
        -------
        array_like
            Returns the value or values of the energy evaluated at r.

        """
        r,u,s = self._zeros(r)
        u = self._interpolate(params)(r)
        if s:
            u = u.item()
        return u

    def _force(self, r, **params):
        """Evaluates the spline force.

        Parameters
        ----------
        r : array_like
            The value or values of r at which to evaluate the energy.
        params : dict
            The parameters (r and knot values) for the potential function.

        Returns
        -------
        array_like
            Returns the value or values of the force evaluated at r.

        """
        r,f,s = self._zeros(r)
        f = -self._interpolate(params).derivative(r, 1)
        if s:
            f = f.item()
        return f

    def derivative(self, pair, var, r):
        #Extending PairPotential method to check if r and knot values are DesignVariables.
        for ri,ki in self.knots(pair):
            if not isinstance(ri, variable.DesignVariable):
                raise TypeError('All r values must be DesignVariables')
            if not isinstance(ki, variable.DesignVariable):
                raise TypeError('All knot values must be DesignVariables')
        return super().derivative(pair, var, r)

    def _derivative(self, param, r, **params):
        """Evaluates the spline parameter derivative using finite differencing.

        Parameters
        ----------
        param : str
            The name of the knot with respect to which to calculate the derivative.
        r : array_like
            The value or values of r at which to evaluate the energy.
        params : dict
            The parameters (r and knot values) for the potential function.

        Returns
        -------
        array_like
            Returns the value or values of the derivative evaluated at r.

        Raises
        ------
        ValueError
            If the parameter with respect to which to take the derivative is not a knot value.

        """
        if 'knot' not in param:
            raise ValueError('Parameter derivative can only be taken for knot values')

        r,d,s = self._zeros(r)
        h = 0.001

        #perturb knot param value
        knot_p = params[param]
        params[param] = knot_p + h
        f_high = self._interpolate(params)(r)
        params[param] = knot_p - h
        f_low = self._interpolate(params)(r)

        params[param] = knot_p
        d = (f_high - f_low)/(2*h)
        if s:
            d = d.item()
        return d

    def _interpolate(self, params):
        """Interpolates the knot points into a continuous spline potential.

        Parameters
        ----------
        params : array_like
            The array of knot values.

        Returns
        -------
        :py:class:`Interpolator`
            The interpolated spline potential.

        """
        r = np.zeros(self.num_knots)
        u = np.zeros(self.num_knots)
        for i in range(self.num_knots):
            ri,ki = self._knot_params(i)
            r[i] = params[ri]
            u[i] = params[ki]
        # reconstruct the energies from differences, starting from the end of the potential
        if self.mode == 'diff':
            u = np.flip(np.cumsum(np.flip(u)))
        return _math.Interpolator(x=r, y=u)

    @property
    def num_knots(self):
        """int: Number of knots."""
        return self._num_knots

    @property
    def mode(self):
        """str: Spline construction mode."""
        return self._mode

    def knots(self, pair):
        """Generator for knot points.

        Parameters
        ----------
        pair : tuple
            The type pair (i,j) for which to retrieve the knot points.

        Yields
        ------
        :py:class:`DesignVariable`
            The next r key in the coefficient array of r values.
        :py:class:`DesignVariable`
            The next knot key in the coefficient array of k values.

        """
        for i in range(self.num_knots):
            ri,ki = self._knot_params(i)
            r = self.coeff[pair][ri]
            k = self.coeff[pair][ki]
            yield r,k

class Yukawa(PairPotential):
    r"""Yukawa pair potential.

    .. math::

        u(r) = \varepsilon\frac{e^{-\kappa r}}{r}

    The required coefficients per pair are:

    - :math:`\varepsilon` (``epsilon``): prefactor (dimensions: energy x length)
    - :math:`\kappa` (``kappa``): inverse screening length

    The optional coefficients per pair are:

    - :math:`r_{\rm min}` (``rmin``): minimum radius, energy and force are 0 for
      :math:`r < r_{\rm min}`. Ignored if ``False`` (default).
    - :math:`r_{\rm max}` (``rmax``): maximum radius, energy and force are 0 for
      :math:`r > r_{\rm max}`. Ignored if ``False`` (default).
    - ``shift``: If ``True``, shift potential to zero at ``rmax`` (default is ``False``).

    Parameters
    ----------
    types : array_like
        List of types (A type must be a `str`).

    """
    def __init__(self, types):
        super().__init__(types=types, params=('epsilon','kappa'))

    def _energy(self, r, epsilon, kappa, **params):
        r"""Evaluates the Yukawa potential energy.

        Parameters
        ----------
        r : array_like
            The value or values of r at which to evaluate the energy.
        epsilon : float or int
            The epsilon parameter for the potential function.
        kappa : float or int
            The kappa parameter for the potential function.

        Returns
        -------
        array_like
            Returns the value or values of the energy evaluated at r.

        Raises
        ------
        ValueError
            If kappa is not a positive number.

        """
        if kappa < 0:
            raise ValueError('kappa must be positive')
        r,u,s = self._zeros(r)
        flags = ~np.isclose(r,0)

        # evaluate potential
        u[flags] = epsilon*np.exp(-kappa*r[flags])/r[flags]
        u[~flags] = np.inf

        if s:
            u = u.item()
        return u

    def _force(self, r, epsilon, kappa, **params):
        r"""Evaluates the Yukawa force.

        Parameters
        ----------
        r : array_like
            The value or values of r at which to evaluate the force.
        epsilon : float or int
            The epsilon parameter for the potential function.
        kappa : float or int
            The kappa parameter for the potential function.

        Returns
        -------
        array_like
            Returns the value or values of the force evaluated at r.

        Raises
        ------
        ValueError
            If kappa is not a positive number.

        """
        if kappa < 0:
            raise ValueError('kappa must be positive')
        r,f,s = self._zeros(r)
        flags = ~np.isclose(r,0)

        # evaluate force
        f[flags] = epsilon*np.exp(-kappa*r[flags])*(1.+kappa*r[flags])/r[flags]**2
        f[~flags] = np.inf

        if s:
            f = f.item()
        return f

    def _derivative(self, param, r, epsilon, kappa, **params):
        r"""Evaluates the Yukawa parameter derivative.

        Parameters
        ----------
        param : `str`
            The parameter with respect to which to take the derivative.
        r : array_like
            The value or values of r at which to evaluate the derivative.
        epsilon : float or int
            The epsilon parameter for the potential function.
        kappa : float or int
            The kappa parameter for the potential function.

        Returns
        -------
        array_like
            Returns the value or values of the derivative evaluated at r.

        Raises
        ------
        ValueError
            If kappa is not a positive number.
        ValueError
            If the parameter with respect to which to take the derivative is not 'kappa' or 'epsilon'.

        """
        if kappa < 0:
            raise ValueError('kappa must be positive')
        r,d,s = self._zeros(r)
        flags = ~np.isclose(r,0)

        # evaluate derivative
        if param == 'epsilon':
            d[flags] = np.exp(-kappa*r[flags])/r[flags]
            d[~flags] = np.inf
        elif param == 'kappa':
            d = -epsilon*np.exp(-kappa*r)
        else:
            raise ValueError('The Yukawa parameters are kappa and epsilon.')

        if s:
            d = d.item()
        return d

class Depletion(PairPotential):
    r"""Depletion pair potential.

    .. math::

        u(r) = -\frac{\pi P(\frac{1}{2}(\sigma_i+\sigma_j)+\sigma_d-r)^2(r^2+r(\sigma_i+\sigma_j+2\sigma_d)-\frac{3}{4}(\sigma_i-\sigma_j)^2}{12r}

    The required coefficients per pair are:

    - :math:`P`: depletant osmotic pressure
    - :math:`\sigma_i`: particle type `i` diameter
    - :math:`\sigma_j`: particle type `j` diameter
    - :math:`\sigma_d`: depletant diameter

    The optional coefficients per pair are:

    - ``rmin``: minimum radius, energy and force are 0 for ``r < rmin``. Ignored if ``False`` (default).
    - ``rmax``: maximum radius, energy and force are 0 for ``r > rmax`` Ignored if ``False`` (default).
    - ``shift``: If ``True``, shift potential to zero at ``rmax`` (default is ``False``).

    Parameters
    ----------
    types : array_like
        List of types (A type must be a str).

    """

    class Cutoff(variable.DependentVariable):
        r"""Dependent variable for the "high bound" or cutoff of the depletion pair potential.

        .. math::

            cutoff = \frac{1}{2}(\sigma_i+\sigma_j)+\sigma_d

        Parameters
        ----------
        sigma_i : int/float or :py:class:`Variable`
            The `sigma_i` parameter of the cutoff.
        sigma_j : int/float or :py:class:`Variable`
            The `sigma_j` parameter of the cutoff.
        sigma_d : int/float or :py:class:`Variable`
            The `sigma_d` parameter of the cutoff.

        """
        def __init__(self, sigma_i, sigma_j, sigma_d):
            super().__init__(sigma_i=sigma_i, sigma_j=sigma_j, sigma_d=sigma_d)

        @property
        def value(self):
            return 0.5*(self.sigma_i.value + self.sigma_j.value) + self.sigma_d.value

        def _derivative(self, param):
            """Calculates the derivative of the Cutoff object with respect to its parameters.

            Parameters
            ----------
            param : str
                The parameter with respect to which to take the derivative.
                (Can only be 'sigma_i', 'sigma_j', or 'sigma_d').

            Returns
            -------
            float
                The calculated derivative value.

            Raises
            ------
            ValueError
                If the parameter argument is not 'sigma_i', 'sigma_j', or 'sigma_d'.

            """
            if param == 'sigma_i':
                return 0.5
            elif param == 'sigma_j':
                return 0.5
            elif param == 'sigma_d':
                return 1.0
            else:
                raise ValueError('Unknown parameter')

    def __init__(self, types, shift=False):
        super().__init__(types=types, params=('P','sigma_i','sigma_j','sigma_d'))

    def energy(self, pair, r):
        # Override parent method to set rmax as cutoff
        if self.coeff[pair]['rmax'] is False:
            self.coeff[pair]['rmax'] = self.Cutoff(self.coeff[pair]['sigma_i'],
                                                   self.coeff[pair]['sigma_j'],
                                                   self.coeff[pair]['sigma_d'])
        return super().energy(pair, r)

    def _energy(self, r, P, sigma_i, sigma_j, sigma_d, **params):
        """Evaluates the depletion potential energy.

        Parameters
        ----------
        r : array_like
            The value or values of r at which to evaluate the energy.
        P : float or int
            The P parameter for the potential function.
        sigma_i : float or int
            The sigma_i parameter for the potential function.
        sigma_j : float or int
            The sigma_j parameter for the potential function.
        sigma_d : float or int
            The sigma_d parameter for the potential function.

        Returns
        -------
        array_like
            Returns the value or values of the energy evaluated at r.

        Raises
        ------
        ValueError
            If sigma_i, sigma_j, and sigma_d are not all positive.

        """
        if sigma_i<=0 or sigma_j<=0 or sigma_d<=0:
            raise ValueError('sigma_i, sigma_j, and sigma_d must all be positive')
        r,u,s = self._zeros(r)

        p1 = (0.5*(sigma_i + sigma_j) + sigma_d - r)**2
        p2 = r**2 + r*(sigma_i + sigma_j + 2.*sigma_d) - 0.75*(sigma_i - sigma_j)**2
        u = -(np.pi*P*p1*p2)/(12.*r)

        if s:
            u = u.item()
        return u

    def force(self, pair, r):
        # Override parent method to set rmax as cutoff
        if self.coeff[pair]['rmax'] is False:
            self.coeff[pair]['rmax'] = self.Cutoff(self.coeff[pair]['sigma_i'],
                                                   self.coeff[pair]['sigma_j'],
                                                   self.coeff[pair]['sigma_d'])
        return super().force(pair, r)

    def _force(self, r, P, sigma_i, sigma_j, sigma_d, **params):
        """Evaluates the depletion force.

        Parameters
        ----------
        r : array_like
            The value or values of r at which to evaluate the force.
        P : float or int
            The P parameter for the potential function.
        sigma_i : float or int
            The sigma_i parameter for the potential function.
        sigma_j : float or int
            The sigma_j parameter for the potential function.
        sigma_d : float or int
            The sigma_d parameter for the potential function.

        Returns
        -------
        array_like
            Returns the value or values of the force evaluated at r.

        Raises
        ------
        ValueError
            If sigma_i, sigma_j, and sigma_d are not all positive.

        """
        if sigma_i<=0 or sigma_j<=0 or sigma_d<=0:
            raise ValueError('sigma_i, sigma_j, and sigma_d must all be positive')
        r,f,s = self._zeros(r)

        p1 = r**2 - 0.25*(sigma_i - sigma_j)**2
        p2 = (0.5*(sigma_i + sigma_j) + sigma_d)**2 - r**2
        f = -(np.pi*P*p1*p2)/(4.*r**2)

        if s:
            f = f.item()
        return f

    def derivative(self, pair, var, r):
        # Override parent method to set rmax as cutoff
        if self.coeff[pair]['rmax'] is False:
            self.coeff[pair]['rmax'] = self.Cutoff(self.coeff[pair]['sigma_i'],
                                                   self.coeff[pair]['sigma_j'],
                                                   self.coeff[pair]['sigma_d'])
        return super().derivative(pair, var, r)

    def _derivative(self, param, r, P, sigma_i, sigma_j, sigma_d, **params):
        """Evaluates the depletion parameter derivative.

        Parameters
        ----------
        param : `str`
            The parameter with respect to which to take the derivative.
        r : array_like
            The value or values of r at which to evaluate the derivative.
        P : float or int
            The P parameter for the potential function.
        sigma_i : float or int
            The sigma_i parameter for the potential function.
        sigma_j : float or int
            The sigma_j parameter for the potential function.
        sigma_d : float or int
            The sigma_d parameter for the potential function.

        Returns
        -------
        array_like
            Returns the value or values of the derivative evaluated at r.

        Raises
        ------
        ValueError
            If sigma_i, sigma_j, and sigma_d are not all positive.
        ValueError
            If the parameter with respect to which to take the derivative
            is not P, sigma_i, sigma_j, or sigma_d.

        """
        if sigma_i<=0 or sigma_j<=0 or sigma_d<=0:
            raise ValueError('sigma_i, sigma_j, and sigma_d must all be positive')
        r,d,s = self._zeros(r)

        if param == 'P':
            p1 = (0.5*(sigma_i + sigma_j) + sigma_d - r)**2
            p2 = r**2 + r*(sigma_i + sigma_j + 2.*sigma_d) - 0.75*(sigma_i - sigma_j)**2
            d = -(np.pi*p1*p2)/(12.*r)
        elif param == 'sigma_i':
            p1 = ((0.5*(sigma_i + sigma_j) + sigma_d - r)
                 *(r**2 + r*(sigma_i + sigma_j + 2.*sigma_d) - 0.75*(sigma_i - sigma_j)**2))
            p2 = (r + 1.5*(sigma_j - sigma_i))*(0.5*(sigma_i + sigma_j) + sigma_d - r)**2
            d = -(np.pi*P*(p1 + p2))/(12.*r)
        elif param == 'sigma_j':
            p1 = ((0.5*(sigma_i + sigma_j) + sigma_d - r)
                 *(r**2 + r*(sigma_i + sigma_j + 2.*sigma_d) - 0.75*(sigma_i - sigma_j)**2))
            p2 = (r + 1.5*(sigma_i - sigma_j))*(0.5*(sigma_i + sigma_j) + sigma_d - r)**2
            d = -(np.pi*P*(p1 + p2))/(12.*r)
        elif param == 'sigma_d':
            p1 = ((sigma_i + sigma_j + 2.*sigma_d - 2.*r)
                 *(r**2 + r*(sigma_i + sigma_j + 2.*sigma_d) - 0.75*(sigma_i - sigma_j)**2))
            p2 = 2.*r*(0.5*(sigma_i + sigma_j) + sigma_d - r)**2
            d = -(np.pi*P*(p1 + p2))/(12.*r)
        else:
            raise ValueError('The depletion parameters are P, sigma_i, sigma_j, and sigma_d.')

        if s:
            d = d.item()
        return d
