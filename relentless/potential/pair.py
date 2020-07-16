__all__ = ['LennardJones']

import numpy as np
from scipy.interpolate import Akima1DInterpolator

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

class Yukawa(PairPotential):
    """Yukawa pair potential.

    .. math::

    u(r) = \varepsilon\frac{{e}^{{-\kappa{r}}}}{{r}}

    The required coefficients per pair are:

    - :math:`\varepsilon`: interaction energy scale
    - :math:`\kappa`: screening length inverse

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

"""TODO: these potentials need updating
class AkimaSpline(PairPotential):
    def __init__(self, types, num_knots, shift=False):
        assert num_knots >= 2, "Must have at least 2 knots"

        super().__init__(types=types,
                         params=['rmin','rmax']+['knot-{}'.format(i) for i in range(num_knots)],
                         default={'rmin': 0.},
                         shift=shift)
        self.num_knots = num_knots

        # N-1 knots are free for optimization
        for pair in self.variables:
            self.variables[pair] = [Variable('knot-{}'.format(i)) for i in range(self.num_knots-1)]

    def energy(self, r, **params):
        r,u = self._zeros(r)
        range_flags = np.logical_and(r >= params['rmin'], r <= params['rmax'])

        akima = self._interpolate(params)
        u[range_flags] = akima(r[range_flags])

        return u

    def force(self, r, pair):
        # load parameters for the pair
        params = self.coeff.evaluate(pair)

        r,f = self._zeros(r)
        range_flags = np.logical_and(r >= params['rmin'], r <= params['rmax'])

        akima = self._interpolate(params)
        f[range_flags] = -akima(r[range_flags], nu=1)

        return f

    def knots(self, pair):
        params = self.coeff.evaluate(pair)
        return self._knots(params)

    def interpolate(self, pair):
        params = self.coeff.evaluate(pair)
        return self._interpolate(params)

    def _knots(self, params):
        r = np.linspace(params['rmin'], params['rmax'], self.num_knots)
        u = [params['knot-{}'.format(i)] for i in range(self.num_knots)]
        return r,u

    def _interpolate(self, params):
        rknot,uknot = self._knots(params)
        return Akima1DInterpolator(rknot, uknot)
"""
