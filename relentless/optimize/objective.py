__all__ = ['RelativeEntropy']

import numpy as np
import scipy.integrate

from .core import OptimizationProblem
from relentless.core import Interpolator,Ensemble

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
                dudp = Interpolator(r,dudp)

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
