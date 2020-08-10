__all__ = ['Depletion','LennardJones','Spline','Yukawa']
import numpy as np

from relentless import core
from .potential import PairPotential

class LennardJones(PairPotential):
    """Lennard-Jones 12-6 pair potential.

    .. math::

        u(r) = 4 \varepsilon\left[\left(\frac{\sigma}{r}\right)^12 - \left(\frac{\sigma}{r}\right)^12 \right]

    The required coefficients per pair are:

    - :math:`\varepsilon`: interaction energy
    - :math:`\sigma`: interaction length scale (e.g., particle diameter)

    The optional coefficients per pair are:

    - ``rmin``: minimum radius, energy and force are 0 for ``r < rmin``. Ignored if ``False`` (default).
    - ``rmax``: maximum radius, energy and force are 0 for ``r > rmax`` Ignored if ``False`` (default).
    - ``shift``: If ``True``, shift potential to zero at ``rmax`` (default is ``False``).

    Setting ``rmax = sigma*2**(1./6.)`` and ``shift = True`` will give the purely repulsive
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
        rs = np.zeros_like(r, dtype=np.float64)
        ks = np.zeros_like(r, dtype=np.float64)
        for i,(ri,ki) in enumerate(zip(r,u)):
            rs[i] = ri.value if isinstance(ri, core.Variable) else ri
            ks[i] = ki.value if isinstance(ki, core.Variable) else ki
        if self.mode == 'diff':
            # difference is next knot minus my knot, with last knot fixed at its current value
            ks[:-1] -= ks[1:]

        # make all knots variables, but hold all r and the last knot const
        for i in range(self.num_knots):
            ri,ki = self._knot_params(i)
            if self.coeff[pair][ri] is None:
                self.coeff[pair][ri] = core.DesignVariable(rs[i],const=True)
            else:
                self.coeff[pair][ri].value = rs[i]
            if self.coeff[pair][ki] is None:
                self.coeff[pair][ki] = core.DesignVariable(ks[i],const=(i==self.num_knots-1))
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

    def derivative(self, pair, param, r):
        #Extending PairPotential method to check if r and knot values are DesignVariables.
        for ri,ki in self.knots(pair):
            if not isinstance(ri, core.DesignVariable):
                raise TypeError('All r values must be DesignVariables')
            if not isinstance(ki, core.DesignVariable):
                raise TypeError('All knot values must be DesignVariables')
        return super().derivative(pair, param, r)

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
        return core.Interpolator(x=r, y=u)

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
    """Yukawa pair potential.

    .. math::

        u(r) = \varepsilon\frac{e^{-\kappa r}}{r}

    The required coefficients per pair are:

    - :math:`\varepsilon`: prefactor (dimensions: energy x length)
    - :math:`\kappa`: inverse screening length

    The optional coefficients per pair are:

    - ``rmin``: minimum radius, energy and force are 0 for ``r < rmin``. Ignored if ``False`` (default).
    - ``rmax``: maximum radius, energy and force are 0 for ``r > rmax`` Ignored if ``False`` (default).
    - ``shift``: If ``True``, shift potential to zero at ``rmax`` (default is ``False``).

    Parameters
    ----------
    types : array_like
        List of types (A type must be a `str`).

    """
    def __init__(self, types, shift=False):
        super().__init__(types=types, params=('epsilon','kappa'))

    def _energy(self, r, epsilon, kappa, **params):
        """Evaluates the Yukawa potential energy.

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
        """Evaluates the Yukawa force.

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
        """Evaluates the Yukawa parameter derivative.

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
    def __init__(self, types, shift=False):
        super().__init__(types=types, params=('P','sigma_i','sigma_j','sigma_d'))

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

        #clamp lo
        lo = 0.5*(sigma_i + sigma_j)
        u_lo = -np.pi*P*sigma_d**3/6.*(1. + 3.*sigma_i*sigma_j/(sigma_d*(sigma_i + sigma_j)))
        f_lo  = self._force(r=lo, P=P, sigma_i=sigma_i, sigma_j=sigma_j, sigma_d=sigma_d)
        below = r < lo
        u[below] = u_lo - f_lo*(r[below] - lo)

        #clamp hi
        hi = lo + sigma_d
        above = r > hi
        u[above] = 0.

        # evaluate in between
        flags = ~np.logical_or(below, above)
        p1 = (0.5*(sigma_i + sigma_j) + sigma_d - r[flags])**2
        p2 = r[flags]**2 + r[flags]*(sigma_i + sigma_j + 2.*sigma_d) - 0.75*(sigma_i - sigma_j)**2
        u[flags] = -(np.pi*P*p1*p2)/(12.*r[flags])

        if s:
            u = u.item()
        return u

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
        lo = 0.5*(sigma_i + sigma_j)
        hi = lo + sigma_d

        #clamp lo
        lo = 0.5*(sigma_i + sigma_j)
        below = r < lo
        f[below] = (-np.pi*P*sigma_i*sigma_j*sigma_d*(sigma_i + sigma_j + sigma_d)
                   /(sigma_i + sigma_j)**2)

        #clamp hi
        hi = lo + sigma_d
        above = r > hi
        f[above] = 0.

        # evaluate in between
        flags = ~np.logical_or(below, above)
        p1 = r[flags]**2 - 0.25*(sigma_i - sigma_j)**2
        p2 = (0.5*(sigma_i + sigma_j) + sigma_d)**2 - r[flags]**2
        f[flags] = -(np.pi*P*p1*p2)/(4.*r[flags]**2)

        if s:
            f = f.item()
        return f

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

        #clamp lo
        lo = 0.5*(sigma_i + sigma_j)
        below = r < lo
        f_lo  = self._force(r=lo, P=P, sigma_i=sigma_i, sigma_j=sigma_j, sigma_d=sigma_d)
        if param == 'P':
            du_lo = -np.pi*sigma_d**3/6.*(1. + 3.*sigma_i*sigma_j/(sigma_d*(sigma_i + sigma_j)))
            df_lo = (-np.pi*sigma_i*sigma_j*sigma_d*(sigma_i + sigma_j + sigma_d)
                    /(sigma_i + sigma_j)**2)
            d_lo = 0.
        elif param == 'sigma_i':
            du_lo = -np.pi*P/2.*(sigma_d*sigma_j/(sigma_i + sigma_j))**2
            df_lo = (-np.pi*P*sigma_d*sigma_j*(sigma_i*(sigma_j - sigma_d) + sigma_j*(sigma_j + sigma_d))
                    /(sigma_i + sigma_j)**3)
            d_lo = 0.5
        elif param == 'sigma_j':
            du_lo = -np.pi*P/2.*(sigma_d*sigma_i/(sigma_i + sigma_j))**2
            df_lo = (-np.pi*P*sigma_d*sigma_i*(sigma_j*(sigma_i - sigma_d) + sigma_i*(sigma_i + sigma_d))
                    /(sigma_i + sigma_j)**3)
            d_lo = 0.5
        elif param == 'sigma_d':
            du_lo = (-np.pi*P*sigma_d*(sigma_d*(sigma_i + sigma_j) + 2.*sigma_i*sigma_j)
                    /(2.*(sigma_i + sigma_j)))
            df_lo = -np.pi*P*sigma_i*sigma_j*(sigma_i + sigma_j + 2.*sigma_d)/(sigma_i + sigma_j)**2
            d_lo = 0.
        else:
            raise ValueError('The depletion parameters are P, sigma_i, sigma_j, and sigma_d.')
        d[below] = du_lo - df_lo*(r[below] - lo) + f_lo*d_lo

        #clamp hi
        hi = lo + sigma_d
        above = r > hi
        d[above] = 0.

        # evaluate in between
        flags = ~np.logical_or(below, above)
        if param == 'P':
            p1 = (0.5*(sigma_i + sigma_j) + sigma_d - r[flags])**2
            p2 = r[flags]**2 + r[flags]*(sigma_i + sigma_j + 2.*sigma_d) - 0.75*(sigma_i - sigma_j)**2
            d[flags] = -(np.pi*p1*p2)/(12.*r[flags])
        elif param == 'sigma_i':
            p1 = ((0.5*(sigma_i + sigma_j) + sigma_d - r[flags])
                 *(r[flags]**2 + r[flags]*(sigma_i + sigma_j + 2.*sigma_d) - 0.75*(sigma_i - sigma_j)**2))
            p2 = (r[flags] + 1.5*(sigma_j - sigma_i))*(0.5*(sigma_i + sigma_j) + sigma_d - r[flags])**2
            d[flags] = -(np.pi*P*(p1 + p2))/(12.*r[flags])
        elif param == 'sigma_j':
            p1 = ((0.5*(sigma_i + sigma_j) + sigma_d - r[flags])
                 *(r[flags]**2 + r[flags]*(sigma_i + sigma_j + 2.*sigma_d) - 0.75*(sigma_i - sigma_j)**2))
            p2 = (r[flags] + 1.5*(sigma_i - sigma_j))*(0.5*(sigma_i + sigma_j) + sigma_d - r[flags])**2
            d[flags] = -(np.pi*P*(p1 + p2))/(12.*r[flags])
        elif param == 'sigma_d':
            p1 = ((sigma_i + sigma_j + 2.*sigma_d - 2.*r[flags])
                 *(r[flags]**2 + r[flags]*(sigma_i + sigma_j + 2.*sigma_d) - 0.75*(sigma_i - sigma_j)**2))
            p2 = 2.*r[flags]*(0.5*(sigma_i + sigma_j) + sigma_d - r[flags])**2
            d[flags] = -(np.pi*P*(p1 + p2))/(12.*r[flags])
        else:
            raise ValueError('The depletion parameters are P, sigma_i, sigma_j, and sigma_d.')

        if s:
            d = d.item()
        return d
