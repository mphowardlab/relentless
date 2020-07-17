__all__ = ['LennardJones','Spline','Yukawa']

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
    valid_modes = ('value','diff')

    def __init__(self, types, num_knots, mode='diff'):
        if isinstance(num_knots,int) and num_knots >= 2:
            self._num_knots = num_knots
        else:
            raise ValueError('Number of spline knots must be an integer >= 2.')

        if mode in self.valid_modes:
            self._mode = mode
        else:
            raise ValueError('Invalid parameter mode, choose from: ' + ','.join(self.modes))

        super().__init__(types=types,
                         params=['r-{}'.format(i) for i in range(num_knots)] + ['knot-{}'.format(i) for i in range(num_knots)])

    def from_array(self, pair, r, u):
        """Setup the potential from arrays of knot positions and energies."""
        # check that r and u have the right shape
        if len(r) != self.num_knots:
            pass
        if len(u) != self.num_knots:
            pass

        # convert to r,knot form given the mode
        rs = np.asarray(r, dtype=np.float64)
        ks = np.asarray(u, dtype=np.float64)
        if mode == 'diff':
            # difference is next knot minus my knot, with last knot fixed at its current value
            ks[:-1] -= ks[1:]

        # make all knots variables, but hold all r and the last knot const
        for i in self.num_knots:
            self.coeff[pair]['r-{}'.format(i)] = core.Variable(rs[i],const=True)
            self.coeff[pair]['knot-{}'.format(i)] = core.Variable(ks[i],const=(i==self.num_knots-1))

    def _energy(self, r, **params):
        r,u,s = self._zeros(r)
        u = self._interpolate(params)(r)
        if s:
            u = u.item()
        return u

    def _force(self, r, **params):
        r,f,s = self._zeros(r)
        f = -self._interpolate(params).derivative(r)
        if s:
            f = f.item()
        return f

    def _derivative(self, param, r, **params):
        raise NotImplementedError()

    def _interpolate(self, params):
        r = np.array([params['r-{}'.format(i)] for i in range(self.num_knots)])
        u = np.array([params['knot-{}'.format(i)] for i in range(self.num_knots)])
        # reconstruct the energies from differences, starting from the end of the potential
        if self.mode == 'diff':
            u = np.flip(np.cumsum(np.flip(u)))
        return core.Interpolator(r, u)

    @property
    def num_knots(self):
        """int: Number of knots."""
        return self._num_knots

    @property
    def mode(self):
        """str: Spline construction mode."""
        return self._mode

    def knots(self, pair):
        """Generator for knot points."""
        for i in range(self.num_knots):
            r = self.coeff[pair]['r-{}'.format(i)]
            k = self.coeff[pair]['knot-{}'.format(i)]
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
