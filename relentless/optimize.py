import json
import os

import numpy as np
import scipy.integrate

from . import core
from .ensemble import Ensemble

class Optimizer:
    def __init__(self, problem):
        self.problem = problem
        self.step = 0

    def run(self):
        raise NotImplementedError()

    def restart(self, env, step):
        """ Restart a calculation from a given step."""
        self.problem.restart(env, step)
        self.step = step

class OptimizationProblem:
    def __init__(self, engine):
        self.engine = engine
        self.step = 0

    def grad(self, env, step):
        raise NotImplementedError()

    def error(self, env, step):
        raise NotImplementedError()

    def restart(self, env, step):
        """ Restart a calculation from a given step."""
        with env.data(step):
            for pot in self.engine.potentials:
                pot.load(step)

    def get_variables(self):
        # flatten down variables for all potentials
        variables = []
        keys = set()
        for pot in self.engine.potentials:
            for key in pot.variables:
                if pot.variables[key] is not None:
                    for var in pot.variables[key]:
                        variables.append((pot,key,var))
                        keys.add(key)

        return variables,keys

class RelativeEntropy(OptimizationProblem):
    def __init__(self, engine, target, dr=0.01):
        super().__init__(engine)
        self.target = target
        self.dr = dr

    def _get_step(self, env, step):
        # try loading the simulation
        try:
            with env.data(step):
                thermo = Ensemble.load('ensemble')
        except FileNotFoundError:
            thermo = None

        # if not found, run the simulation
        if thermo is None:
            # write the current parameters
            with env.data(step):
                for pot in self.engine.potentials:
                    pot.save()

            self.engine.run(env, step)
            thermo = self.engine.process(env, step)
            with env.data(step):
                thermo.save('ensemble')

        return thermo

    def grad(self, env, step):
        thermo = self._get_step(env, step)

        # compute the gradient
        gtgt = self.target.rdf
        gsim = thermo.rdf
        gradient = {}
        variables,_ = self.get_variables()
        for pot,key,param in variables:
            # sum derivative over all gij
            update = 0.
            for i,j in pot.coeff:
                # compute derivative and interpolate through r
                r0 = max(gsim[i,j].domain[0], gtgt[i,j].domain[0])
                r1 = min(gsim[i,j].domain[1], gtgt[i,j].domain[1])
                r = np.arange(r0,r1+0.5*self.dr,self.dr)
                dudp = pot.derivative(r, (i,j), key, param.name)
                dudp = core.Interpolator(r,dudp)

                # take integral by trapezoidal rule
                sim_factor = thermo.N[i]*thermo.N[j]/thermo.V
                tgt_factor = self.target.N[i]*self.target.N[j]/self.target.V
                mult = 2 if i == j else 4 # 2 if same, otherwise need i,j and j,i contributions
                update += scipy.integrate.trapz(x=r, y=mult*np.pi*r**2*(sim_factor*gsim[i,j](r)-tgt_factor*gtgt[i,j](r))*self.target.beta*dudp(r))

            gradient[(pot,key,param)] = update
        return gradient

    def error(self, env, step):
        thermo = self._get_step(env,step)

        # compute the error
        gtgt = self.target.rdf
        gsim = thermo.rdf
        diff = 0.
        for i,j in self.target.rdf:
            if self.target.rdf[i,j] is not None:
                # compute derivative and interpolate through r
                r0 = max(gsim[i,j].domain[0], gtgt[i,j].domain[0])
                r1 = min(gsim[i,j].domain[1], gtgt[i,j].domain[1])
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
                    for pot in self.problem.engine.potentials:
                        pot.save()

            self.step += 1

        return converged
