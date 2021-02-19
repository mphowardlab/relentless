import abc

from .objective import ObjectiveFunction

class Optimizer(abc.ABC):
    def __init__(self, abs_tol):
        self.abs_tol = abs_tol

    @abc.abstractmethod
    def optimize(self, objective):
        pass

    def has_converged(self, result):
        for x in result.variables:
            grad = result.gradient(x)
            try:
                tol = self.abs_tol[x]
            except TypeError:
                tol = self.abs_tol
            except KeyError:
                raise KeyError('Absolute tolerance not set for design variable ' + str(x))
            if abs(grad) > tol:
                return False

        return True

    @property
    def abs_tol(self):
        return self._abs_tol

    @abs_tol.setter
    def abs_tol(self, value):
        try:
            self._abs_tol = dict(value)
            err = any([tol < 0 for tol in self._abs_tol.values()])
        except TypeError:
            self._abs_tol = value
            err = self._abs_tol < 0
        if err:
            raise ValueError('Absolute tolerances must be non-negative.')

class SteepestDescent(Optimizer):
    def __init__(self, step_size, max_iter, abs_tol):
        super().__init__(abs_tol)
        self.step_size = step_size
        self.max_iter = max_iter

    def optimize(self, objective):
        dvars = objective.design_variables()
        if len(dvars) == 0:
            return None

        iter_num = 0
        converged = False
        while not converged and iter_num < self.max_iter:
            res = objective.compute()
            for x in dvars:
                x.value -= self.step_size*res.gradient(x)
            converged = self.has_converged(res)
            iter_num += 1

        return converged

    @property
    def step_size(self):
        return self._step_size

    @step_size.setter
    def step_size(self, value):
        if value <= 0:
            raise ValueError('The step size must be positive.')
        self._step_size = value

    @property
    def max_iter(self):
        return self._max_iter

    @max_iter.setter
    def max_iter(self, value):
        if not isinstance(value, int):
            raise TypeError('The maximum number of iterations must be an integer.')
        if value < 1:
            raise ValueError('The maximum number of iterations must be positive.')
        self._max_iter = value
