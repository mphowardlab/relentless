__all__ = ['Optimizer','SteepestDescent']

import abc

from .objective import ObjectiveFunction

class Optimizer(abc.ABC):
    def __init__(self, obj):
        self.objective = obj

    @abc.abstractmethod
    def run(self):
        pass

class SteepestDescent(Optimizer):
    def __init__(self, obj, alpha):
        super().__init__(obj)
        self.alpha = alpha

    def run(self):
        res = self.objective.run()
        for x in res.gradient:
            x.value += self.alpha*res.gradient[x]
