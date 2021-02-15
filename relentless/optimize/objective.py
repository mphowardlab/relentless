import abc

class ObjectiveFunction(abc.ABC):

    class Result:
        def __init__(self, value, gradient):
            self.value = value
            self._gradient = gradient

        def gradient(self, var):
            try:
                return self._gradient[var]
            except KeyError:
                return 0.0

    @abc.abstractmethod
    def compute(self):
        pass

    @abc.abstractmethod
    def design_variables(self):
        pass
