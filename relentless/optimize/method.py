import abc

from .objective import ObjectiveFunction

class Optimizer(abc.ABC):

    @abc.abstractmethod
    def optimize(self, objective):
        pass

class SteepestDescent(Optimizer):
    def __init__(self, alpha, max_iter, abs_tol):
        super().__init__()
        self._alpha = alpha
        self._max_iter = max_iter
        self._abs_tol = abs_tol

    def optimize(self, objective):
        dvars = objective.design_variables()
        if len(dvars) == 0:
            return False

        iter_num = 0
        converged = False
        while not converged and iter_num < self.max_iter:
            res = objective.compute()
            converged = self._has_converged(dvars, res)
            for x in dvars:
                x.value -= self.alpha*res.gradient(x)
            iter_num += 1

        return converged

    def _has_converged(self, dvars, res):
        for x in dvars:
            grad = res.gradient(x)
            try:
                if abs(grad) > self.abs_tol:
                    return False
            except TypeError:
                if abs(grad) > self.abs_tol[x]:
                    return False
        return True

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if value <= 0:
            raise ValueError('alpha must be positive.')
        self._alpha = value

    @property
    def max_iter(self):
        return self._max_iter

    @max_iter.setter
    def max_iter(self, value):
        if not isinstance(value, int):
            raise TypeError('The maximum number of iterations must be an integer.')
        if value <= 0:
            raise ValueError('The maximum number of iterations must be positive.')
        self._max_iter = value

    @property
    def abs_tol(self):
        return self._abs_tol

    @abs_tol.setter
    def abs_tol(self, value):
        try:
            if value <= 0:
                raise ValueError('The absolute tolerance must be positive.')
            self._abs_tol = value
        except TypeError:
            if not isinstance(self._abs_tol, dict):
                self._abs_tol = {}
            for x in value:
                if value[x] <= 0:
                    raise ValueError('The absolute tolerance for ' + str(x) + ' must be positive.')
                self._abs_tol[x] = value[x]
