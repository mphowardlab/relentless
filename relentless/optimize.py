import json
import os

import numpy as np
import scipy.integrate

from . import core
from . import rdf

class Optimizer(object):
    def __init__(self, problem):
        self.problem = problem
        self.step = 0

    def run(self):
        raise NotImplementedError()

    def restart(self, step):
        """ Restart a calculation from a given step."""
        self.problem.restart(step)
        self.step = step

class OptimizationProblem(object):
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

    def grad(self, env):
        raise NotImplementedError()

    def restart(self, step):
        """ Restart a calculation from a given step."""
        for pot in self.potentials:
            pot.load(step)

    def get_variables(self):
        # flatten down variables for all potentials
        variables = []
        keys = set()
        for pot in self.potentials:
            for key in pot.variables:
                if pot.variables[key] is not None:
                    for var in pot.variables[key]:
                        variables.append((pot,key,var))
                        keys.add(key)

        return variables,keys

class RelativeEntropy(OptimizationProblem):
    def __init__(self, target, engine, rdf, table, dr=0.01):
        super().__init__(engine, rdf, table)
        self.target = target
        self.dr = dr

    def grad(self, env, step):
        variables,keys = self.get_variables()

        # fit the target g(r) to a spline
        # TODO: abstract to RDF class
        gtgt = {}
        for pair in self.target.rdf:
            if self.target.rdf[pair] is not None:
                gtgt[pair] = core.Interpolator(self.target.rdf[pair])

        # create potentials
        potentials = self._tabulate_potentials()

        # run the simulation
        self.engine.run(env, step, self.target, potentials)

        # evaluate the RDFs
        thermo = self.rdf.run(env, step)
        gsim = {}
        for pair in thermo.rdf:
            if thermo.rdf[pair] is not None:
                gsim[pair] = core.Interpolator(thermo.rdf[pair])

        # save the current values
        with env.data(step):
            for i,j in thermo.rdf:
                file_ = 'rdf.{i}.{j}.dat'.format(i=i, j=j)
                np.savetxt(file_, thermo.rdf[i,j], header='r g(r)')

            for pot in self.potentials:
                pot.save()

        # update parameters
        gradient = {}
        for pot,key,param in variables:
            # sum derivative over all gij
            update = 0.
            for i,j in pot.coeff:
                # compute derivative and interpolate through r
                r0 = max(gsim[i,j].rmin, gtgt[i,j].rmin)
                r1 = min(gsim[i,j].rmax, gtgt[i,j].rmax)
                r = np.arange(r0,r1+0.5*self.dr,self.dr)
                dudp = pot.derivative(r, (i,j), key, param.name)
                dudp = core.Interpolator(np.column_stack((r,dudp)))

                # take integral by trapezoidal rule
                sim_factor = thermo.N[i]*thermo.N[j]/thermo.V
                tgt_factor = self.target.N[i]*self.target.N[j]/self.target.V
                mult = 2 if i == j else 4 # 2 if same, otherwise need i,j and j,i contributions?
                update += scipy.integrate.trapz(x=r, y=mult*np.pi*r**2*(sim_factor*gsim[i,j](r)-tgt_factor*gtgt[i,j](r))*self.target.beta*dudp(r))

            gradient[(pot,key,param)] = update
        return gradient

    def error(self, env, step):
        # fit the target g(r) to a spline
        gtgt = {}
        for pair in self.target.rdf:
            if self.target.rdf[pair] is not None:
                gtgt[pair] = core.Interpolator(self.target.rdf[pair])

        # evaluate the RDFs
        thermo = self.rdf.run(env, step)
        gsim = {}
        for pair in thermo.rdf:
            if thermo.rdf[pair] is not None:
                gsim[pair] = core.Interpolator(thermo.rdf[pair])

        diff = 0.
        for i,j in self.target.rdf:
            if self.target.rdf[i,j] is not None:
                # compute derivative and interpolate through r
                r0 = max(gsim[i,j].rmin, gtgt[i,j].rmin)
                r1 = min(gsim[i,j].rmax, gtgt[i,j].rmax)
                r = np.arange(r0,r1+0.5*self.dr,self.dr)

                sim_factor = thermo.N[i]*thermo.N[j]/thermo.V
                tgt_factor = self.target.N[i]*self.target.N[j]/self.target.V
                diff += scipy.integrate.trapz(x=r, y=4.*np.pi*r**2*(sim_factor*gsim[i,j](r)-tgt_factor*gtgt[i,j](r))**2)

        return diff

class GradientDescent(Optimizer):
    def __init__(self, problem):
        super().__init__(problem)

    def run(self, env, rates, maxiter, gradtol=1.e-3, rdftol=1.e-4):
        variables,keys = self.problem.get_variables()

        # check keys are properly set
        for key in keys:
            if key not in rates:
                raise KeyError('Learning rate not set for {},{} pair.'.format(*key))

        converged = False
        while self.step < maxiter and not converged:
            # get gradient
            gradient = self.problem.grad(env, self.step)
            gvec = []
            for (pot,key,param),update in gradient.items():
                param.value = pot.coeff[key][param.name] + rates[key] * update
                gvec.append(update)
            gvec = np.array(gvec)

            # convergence: 2-norm of the gradient
            convergence = {}
            convergence['gradient'] = np.sum(gvec*gvec)

            # convergence: rdf error
            convergence['rdf_diff'] = self.problem.error(env, self.step)

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

            converged = convergence['gradient'] < gradtol or convergence['rdf_diff'] < rdftol or convergence['constraints']
            if not converged:
                # complete update step
                for pot,pair,param in variables:
                    pot.coeff[pair][param.name] = param.value

                # stash new parameters into next step
                with env.data(self.step+1):
                    for pot in self.problem.potentials:
                        pot.save()

            self.step += 1

        return converged
