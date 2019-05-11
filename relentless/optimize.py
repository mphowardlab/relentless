from __future__ import division

import os

import numpy as np
import scipy.integrate
import scipy.interpolate

from . import core
from . import rdf
from . import utils

class Variable(object):
    def __init__(self, name, rate, low=None, high=None):
        self.name = name
        self.rate = rate
        self.low = low
        self.high = high

        self.tmp = None

    def check(self, value):
        return not ((self.low is not None and value < self.low) or
                    (self.high is not None and value > self.high))

    def clamp(self, value):
        if self.low is not None and value < self.low:
            return self.low, True
        elif self.high is not None and value > self.high:
            return self.high, True
        else:
            return value, False

class Optimizer(object):
    def __init__(self, engine, rdf, table):
        self.engine = engine
        self.rdf = rdf
        self.table = table

        self.potentials = set()
        self.step = 0

        self._types = set()
        self._update_types = True

    def add_potential(self, potential):
        self.potentials.add(potential)
        self._update_types = True

    def remove_potential(self, potential):
        self.potentials.remove(potential)
        self._update_types = True

    @property
    def types(self):
        if self._update_types:
            self._types = set()
            for pot in self.potentials:
                for t in pot.coeff.types:
                    self._types.add(t)
            self._update_types = False

        return self._types

    def _tabulate_potentials(self):
        potentials = core.CoefficientMatrix(self.types)
        for pair in potentials.pairs:
            r = self.table.r
            u = self.table(pair, self.potentials)
            f = self.table.force(pair, self.potentials)
            potentials[pair] = self.table.regularize(u, f, trim=self.engine.trim)
        return potentials

    def _compute_thermo(self, env, traj, target):
        ens = target.copy()

        # number of particles per type
        ens.V = 0.
        for i in self.types:
            ens.N[i] = 0

        # average over trajectory
        for s in traj:
            ens.V += s.box.volume
            for i in self.types:
                ens.N[i] += np.sum(s.types == i)
        ens.V /= len(traj)
        for i in self.types:
            ens.N[i] /= len(traj)

        for pair in target.rdf:
            gtgt = target.rdf[pair]
            rmax = self._get_rmax(gtgt[:,0])
            gij = self.rdf(env, traj, pair, rmax)
            ens.rdf[pair] = self.rdf(env, traj, pair, rmax)

        return ens

    def _get_rmax(self, r):
        r0 = r[0]
        dr = r[1] - r0

        rmax = r[-1]
        if np.isclose(r0, 0.):
            # left end
            rmax += dr
        elif np.isclose(r0, 0.5*dr):
            # midpoint
            rmax += 0.5*dr

        return rmax

    def run(self):
        raise NotImplementedError()

    def restart(self, step):
        """ Restart a calculation from a given step."""
        for pot in self.potentials:
            pot.load(step)
        self.step = step

class SteepestDescent(Optimizer):
    def __init__(self, engine, rdf, table):
        super(SteepestDescent, self).__init__(engine, rdf, table)

    def run(self, env, target, maxiter, dr=0.01):
        # fit the target g(r) to a spline
        gtgt = {}
        for i,j in target.rdf:
            gij = target.rdf[i,j]
            gtgt[i,j] = rdf.RDFInterpolator(gij[:,0],gij[:,1])

        # flatten down the unique tunable parameters for all potentials
        free = []
        for pot in self.potentials:
            for a,b in pot.free:
                if b >= a:
                    for param in pot.free[a,b]:
                        free.append((pot,(a,b),param))

        converged = False
        while self.step < maxiter and not converged:
            env.cwd = utils.TemporaryWorkingDirectory(env.scratch(str(self.step)))

            # create potentials
            potentials = self._tabulate_potentials()

            # run the simulation
            self.engine.run(env, self.step, potentials)

            # get the trajectory
            traj = self.engine.load_trajectory(env)

            # evaluate the RDFs
            thermo = self._compute_thermo(env, traj, target)
            gsim = {}
            for i,j in thermo.rdf:
                gij = thermo.rdf[i,j]
                gsim[i,j] = rdf.RDFInterpolator(gij[:,0],gij[:,1])

            # update parameters
            converged = True
            for pot,key,param in free:
                # sum derivative over all gij
                update = 0.
                for i,j in pot.coeff:
                    # compute derivative and interpolate through r
                    r0 = max(gsim[i,j].rmin, gtgt[i,j].rmin)
                    r1 = min(gsim[i,j].rmax, gtgt[i,j].rmax)
                    r = np.arange(r0,r1+0.5*dr,dr)

                    dudp = pot.derivative(r, (i,j), key, param.name)
                    dudp = scipy.interpolate.Akima1DInterpolator(x=r, y=dudp)
                    # take integral by quadrature
                    sim_factor = thermo.N[i]*thermo.N[j]/thermo.V
                    tgt_factor = target.N[i]*target.N[j]/target.V
                    update += scipy.integrate.trapz(x=r, y=2.*np.pi*r**2*(sim_factor*gsim[i,j](r)-tgt_factor*gtgt[i,j](r))*dudp(r))

                if np.abs(update / pot.coeff[key][param.name]) > 1.e-2:
                    converged = False
                param.tmp,_ = param.clamp(pot.coeff[key][param.name] + param.rate * update)

            if not converged:
                # complete update step
                for pot,pair,param in free:
                    pot.coeff[pair][param.name] = param.tmp

                print("{} {}".format(free[0][0].coeff['A','A']['epsilon'],free[0][0].coeff['A','A']['sigma']))
                print("{} {}".format(free[0][0].coeff['B','B']['epsilon'],free[0][0].coeff['B','B']['sigma']))
                print("")

            self.step += 1

            env.reset()
        print(converged)
