import json
import os

import numpy as np
import scipy.integrate

from . import core
from . import rdf

class Optimizer(object):
    def __init__(self, engine, trajectory, rdf, table):
        self.engine = engine
        self.trajectory = trajectory
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
            if target.rdf[pair] is not None:
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
    def __init__(self, engine, trajectory, rdf, table):
        super().__init__(engine, trajectory, rdf, table)

    def run(self, env, target, rates, maxiter, dr=0.01):
        # flatten down variables for all potentials
        variables = []
        keys = set()
        for pot in self.potentials:
            has_vars = False
            for key in pot.variables:
                if pot.variables[key] is not None:
                    has_vars = True
                    for var in pot.variables[key]:
                        variables.append((pot,key,var))
                        keys.add(key)
            if has_vars:
                for pair in pot.coeff.pairs:
                    if pair not in target.rdf or target.rdf[pair] is None:
                        raise KeyError('RDF for {},{} pair required.'.format(*pair))

        # check keys are properly set
        for key in keys:
            if key not in rates:
                raise KeyError('Learning rate not set for {},{} pair.'.format(*key))

        # fit the target g(r) to a spline
        gtgt = {}
        for pair in target.rdf:
            if target.rdf[pair] is not None:
                gtgt[pair] = core.Interpolator(target.rdf[pair])

        converged = False
        while self.step < maxiter and not converged:
            # create potentials
            potentials = self._tabulate_potentials()

            # run the simulation
            self.engine.run(env, self.step, potentials)

            # get the trajectory
            traj = self.trajectory.load(env, self.step)

            # evaluate the RDFs
            thermo = self._compute_thermo(env, traj, target)
            gsim = {}
            for pair in thermo.rdf:
                if thermo.rdf[pair] is not None:
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
            for pot,key,param in variables:
                # sum derivative over all gij
                update = 0.
                for i,j in pot.coeff:
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

                param.value = pot.coeff[key][param.name] + rates[key] * update
                gradient.append(update)
            gradient = np.asarray(gradient)

            # convergence: 2-norm of the gradient
            convergence = {}
            convergence['gradient'] = np.sum(gradient*gradient)

            # convergence: rdf error
            convergence['rdf_diff'] = 0.
            for i,j in target.rdf:
                if target.rdf[i,j] is not None:
                    # compute derivative and interpolate through r
                    r0 = max(gsim[i,j].rmin, gtgt[i,j].rmin)
                    r1 = min(gsim[i,j].rmax, gtgt[i,j].rmax)
                    r = np.arange(r0,r1+0.5*dr,dr)

                    sim_factor = thermo.N[i]*thermo.N[j]/thermo.V
                    tgt_factor = target.N[i]*target.N[j]/target.V
                    convergence['rdf_diff'] += scipy.integrate.trapz(x=r, y=4.*np.pi*r**2*(sim_factor*gsim[i,j](r)-tgt_factor*gtgt[i,j](r))**2)

            # TODO: convergence: contraints
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
                for pot,pair,param in variables:
                    pot.coeff[pair][param.name] = param.value

                # stash new parameters into next step
                with env.data(self.step+1):
                    for pot in self.potentials:
                        pot.save()

            self.step += 1

        return converged
