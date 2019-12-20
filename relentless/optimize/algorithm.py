__all__ = ['GradientDescent']

import os
import numpy as np

from .core import Optimizer

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
