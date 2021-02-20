import abc

from relentless import _collections

class ObjectiveFunction(abc.ABC):

    class Result:
        def __init__(self, value, gradient, objective):
            self.value = value

            # gradient must be keyed only on objective design variables
            self.__gradient = _collections.FixedKeyDict(keys=objective.design_variables())
            self.__gradient.update(gradient)
            self._gradient = self.__gradient.todict()

        def gradient(self, var):
            try:
                return self._gradient[var]
            except KeyError:
                return 0.0

        @property
        def variables(self):
            return self._gradient.keys()

    @abc.abstractmethod
    def compute(self):
        pass

    @abc.abstractmethod
    def design_variables(self):
        pass
