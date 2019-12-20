__all__ = ['LennardJones', 'AkimaSpline']

import numpy as np
from scipy.interpolate import Akima1DInterpolator

from .core import PairPotential

class LennardJones(PairPotential):
    def __init__(self, types, shift=False):
        super().__init__(types=types,
                         params=('epsilon','sigma','rmin','rmax'),
                         default={'rmin': 0},
                         shift=shift)

    def energy(self, r, epsilon, sigma, rmin, rmax):
        r,u = self._zeros(r)
        flags = ~np.isclose(r, 0)

        # evaluate potential for r in range, but nonzero
        r6_inv = np.power(sigma/r[flags], 6)
        u[flags] = 4.*epsilon*(r6_inv**2 - r6_inv)

        # evaluate potential for r in range, but zero
        u[~flags] = np.inf

        return u

    def force(self, r, pair):
        # load parameters for the pair
        params = self.coeff.evaluate(pair)
        epsilon = params['epsilon']
        sigma = params['sigma']

        r,f = self._zeros(r)
        flags = ~np.isclose(r, 0)

        # evaluate force for r in range, but nonzero
        rinv = 1./r[flags]
        r6_inv = np.power(sigma*rinv, 6)
        f[flags] = (48.*epsilon*rinv)*(r6_inv**2-0.5*r6_inv)

        # evaluate force for r in range, but zero
        f[~flags] = np.inf

        return f

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
