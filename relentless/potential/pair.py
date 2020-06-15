__all__ = ['LennardJones']

import numpy as np
from scipy.interpolate import Akima1DInterpolator

from .potential import PairPotential

class LennardJones(PairPotential):
    def __init__(self, types):
        super().__init__(types=types, params=('epsilon','sigma'))

    def _energy(self, r, epsilon, sigma, **params):
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
        r,d,s = self._zeros(r)
        flags = ~np.isclose(r,0)

        # evaluate derivative
        r6_inv = np.power(sigma/r[flags], 6)
        if param == 'epsilon':
            d[flags] = 4*(r6_inv**2 - r6_inv)
        elif param == 'sigma' and sigma > 0:
            d[flags] = (48.*epsilon/sigma)*(r6_inv**2 - 0.5*r6_inv)
        d[~flags] = np.inf

        if s:
            d = d.item()
        return d

""" TODO: these potentials need updating
class Yukawa(PairPotential):
    def __init__(self, types, shift=False):
        super().__init__(types=types, params=('epsilon','kappa'))

    def energy(self, r, epsilon, kappa, **params):
        r,u = self._zeros(r)
        flags = ~np.isclose(r,0)

        u[flags] = epsilon*np.exp(-kappa*r[flags])/r[flags]
        u[~flags] = np.inf

        return u

    def force(self, r, pair):
        # load parameters for the pair
        params = self.coeff.evaluate(pair)
        epsilon = params['epsilon']
        kappa = params['kappa']

        r,f = self._zeros(r)
        flags = ~np.isclose(r,0)

        f[flags] = epsilon*np.exp(-kappa*r[flags])*(1.+kappa*r[flags])/r[flags]**2
        f[~flags] = np.inf

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
