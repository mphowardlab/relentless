from __future__ import division

import json
import os

import numpy as np
import scipy.integrate

from . import core
from . import rdf

class Variable(object):
    def __init__(self, name, value=None, low=None, high=None):
        self.name = name
        self.low = low
        self.high = high

        self._value = value
        self._free = self.check(self._value) if self._value is not None else False

    def check(self, value):
        if self.low is not None and value <= self.low:
            return -1
        elif self.high is not None and value >= self.high:
            return 1
        else:
            return 0

    def clamp(self, value):
        b = self.check(value)
        if b == -1:
            v = self.low
        elif b == 1:
            v = self.high
        else:
            v = value

        return v,b

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        v,b = self.clamp(value)
        self._value = v
        self._free = b

    @property
    def free(self):
        return self._free == 0

    def is_low(self):
        return self._free == -1

    def is_high(self):
        return self._free == 1

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
        potentials = core.PairMatrix(self.types)
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
        self._variables = []
        self.rates = {}

    def add_potential(self, potential, variables, rates):
        super(SteepestDescent, self).add_potential(potential)

        # copy variables
        keys = []
        for pair in variables:
            for var in variables[pair]:
                if pair[1] < pair[0]:
                    key = pair[::-1]
                else:
                    key = pair
                self._variables.append((potential,key,var))
                keys.append(key)

        # copy rate coefficients
        self.rates[potential] = {}
        for pair in rates:
            if pair[1] < pair[0]:
                key = pair[::-1]
            else:
                key = pair
                self.rates[potential][key] = rates[pair]

        # check all keys are in rate
        for key in keys:
            if key not in self.rates[potential]:
                raise KeyError('Missing learning rate for key ({},{}).'.format(*key))

    def remove_potential(self, potential):
        super(SteepestDescent, self).remove_potential(potential)

        # remove variables referencing this potential
        self._variables = [var for var in self._variables if var[0] != potential]

        # remove rate coefficients
        del self.rates[potential]

    def run(self, env, target, maxiter, dr=0.01):
        # fit the target g(r) to a spline
        gtgt = {}
        for pair in target.rdf:
            gtgt[pair] = core.Interpolator(target.rdf[pair])

        converged = False
        while self.step < maxiter and not converged:
            # create potentials
            potentials = self._tabulate_potentials()

            # run the simulation
            self.engine.run(env, self.step, potentials)

            # get the trajectory
            traj = self.engine.load_trajectory(env, self.step)

            # evaluate the RDFs
            thermo = self._compute_thermo(env, traj, target)
            gsim = {}
            for pair in thermo.rdf:
                gsim[pair] = core.Interpolator(thermo.rdf[pair])

            # save the current values
            with env.data(self.step):
                for i,j in thermo.rdf:
                    file_ = 'rdf.{i}.{j}.dat'.format(i=i, j=j)
                    np.savetxt(file_, thermo.rdf[i,j], header='r g(r)')

                for pot in self.potentials:
                    pot.save()

            # update parameters
            gradient = []
            for pot,key,param in self._variables:
                # sum derivative over all gij
                update = 0.
                for i,j in target.rdf:
                    # only operate on pair if present in potential
                    if (i,j) not in pot.coeff:
                        continue

                    # compute derivative and interpolate through r
                    r0 = max(gsim[i,j].rmin, gtgt[i,j].rmin)
                    r1 = min(gsim[i,j].rmax, gtgt[i,j].rmax)
                    r = np.arange(r0,r1+0.5*dr,dr)
                    dudp = pot.derivative(r, (i,j), key, param.name)
                    dudp = core.Interpolator(np.column_stack((r,dudp)))

                    # take integral by trapezoidal rule
                    sim_factor = thermo.N[i]*thermo.N[j]/thermo.V
                    tgt_factor = target.N[i]*target.N[j]/target.V
                    update += scipy.integrate.trapz(x=r, y=2.*np.pi*r**2*(sim_factor*gsim[i,j](r)-tgt_factor*gtgt[i,j](r))*dudp(r))

                param.value = pot.coeff[key][param.name] + self.rates[pot][key] * update
                gradient.append(update)
            gradient = np.asarray(gradient)

            # convergence: 2-norm of the gradient
            convergence = {}
            convergence['gradient'] = np.sum(gradient*gradient)

            # convergence: rdf error
            convergence['rdf_diff'] = 0.
            for i,j in target.rdf:
                # compute derivative and interpolate through r
                r0 = max(gsim[i,j].rmin, gtgt[i,j].rmin)
                r1 = min(gsim[i,j].rmax, gtgt[i,j].rmax)
                r = np.arange(r0,r1+0.5*dr,dr)

                sim_factor = thermo.N[i]*thermo.N[j]/thermo.V
                tgt_factor = target.N[i]*target.N[j]/target.V
                convergence['rdf_diff'] += scipy.integrate.trapz(x=r, y=4.*np.pi*r**2*(sim_factor*gsim[i,j](r)-tgt_factor*gtgt[i,j](r))**2)

            # convergence: contraints
            convergence['constraints'] = True
            for g,(pot,key,param) in zip(gradient, self._variables):
                v = pot.coeff[key][param.name]
                if (g > 0 and v < param.high) or (g < 0 and v > param.low):
                    convergence['constraints'] = False

            with env.project:
                write_header = not os.path.exists('convergence.dat') or self.step == 0
                mode = 'a' if not write_header else 'w'
                with open('convergence.dat',mode) as f:
                    if write_header:
                        f.write('# step |gradient|^2 rdf-error\n')
                    f.write('{step} {grad2} {diff}\n'.format(step=self.step,
                                                           grad2=convergence['gradient'],
                                                           diff=convergence['rdf_diff']))

            converged = convergence['gradient'] < 1.e-3 or convergence['rdf_diff'] < 1.e-4 or convergence['constraints']
            if not converged:
                # complete update step
                for pot,pair,param in self._variables:
                    pot.coeff[pair][param.name] = param.value

                # stash new parameters into next step
                with env.data(self.step+1):
                    for pot in self.potentials:
                        pot.save()

                print("{} {}".format(self._variables[0][0].coeff['A','A']['epsilon'],self._variables[0][0].coeff['A','A']['sigma']))
                print("{} {}".format(self._variables[0][0].coeff['B','B']['epsilon'],self._variables[0][0].coeff['B','B']['sigma']))
                print("")

            self.step += 1

        return converged
