import abc

from .objective import ObjectiveFunction

class Optimizer(abc.ABC):

    @abc.abstractmethod
    def optimize(self, objective):
        pass

class SteepestDescent(Optimizer):
    def __init__(self, alpha):
        self.alpha = alpha

    def optimize(self, objective):
        dvars = objective.design_variables()
        res = objective.compute()
        for x in dvars:
            x.value -= self.alpha*res.gradient(x)
