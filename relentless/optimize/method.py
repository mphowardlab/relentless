import abc

from .objective import ObjectiveFunction

class Optimizer(abc.ABC):

    @abc.abstractmethod
    def optimize(self, objective):
        pass

class SteepestDescent(Optimizer):
    def __init__(self, alpha, max_iter, abs_tol, mode):
        self.alpha = alpha
        self.max_iter = max_iter
        self.abs_tol = abs_tol
        self.mode = mode

    def optimize(self, objective):
        dvars = objective.design_variables()
        iter_num = 1
        update = {}

        while True:
            res = objective.compute()
            for x in dvars:
                update[x] = self.alpha*res.gradient(x)

            if iter_num > self.max_iter:
                raise RuntimeError('Could not find solution within ' + str(self.max_iter) + ' iterations.')
            elif self._has_converged(update):
                print('Solution found in ' + str(iter_num) + ' iterations.')
                break
            #TODO: add this branch?
            #elif alpha*update/value
                # raise ValueError('Converged to wrong minimum.')
            else:
                for x in dvars:
                    x.value -= update[x]

            iter_num += 1

    def _has_converged(self, update):
        if self.mode=='grad_diff':
            for x in update:
                if abs(update[x]) > (self.abs_tol[x] if isinstance(self.abs_tol, dict) else self.abs_tol):
                    return False
            return True
        else:
            raise ValueError('Convergence mode not found.')
