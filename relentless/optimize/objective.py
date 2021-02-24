import abc

from relentless import _collections

class ObjectiveFunction(abc.ABC):

    @abc.abstractmethod
    def compute(self):
        pass

    @abc.abstractmethod
    def design_variables(self):
        pass

    def make_result(self, value, gradient):
        return ObjectiveFunctionResult(value, gradient, self)

class ObjectiveFunctionResult:
    def __init__(self, value, gradient, objective):
        self.value = value
        self._gradient = _collections.FixedKeyDict(keys=objective.design_variables())
        self._gradient.update(gradient)

    def gradient(self, var):
        try:
            return self._gradient[var]
        except KeyError:
            return 0.0

    def design_variables(self):
        return self._gradient.todict().keys()
