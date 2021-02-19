import abc

from relentless import _collections

class ObjectiveFunction(abc.ABC):

    class Result:
        def __init__(self, value, gradient, objective):
            self.value = value

            # gradient must be keyed only on objective design variables
            str_gradient = {}
            lookup = {}
            for k in gradient:
                str_k = str(k)
                str_gradient[str_k] = gradient[k]
                lookup[str_k] = k

            str_dvars = [str(j) for j in objective.design_variables()]
            fk_gradient = _collections.FixedKeyDict(keys=str_dvars)
            fk_gradient.update(str_gradient)

            self._gradient = {}
            for k in fk_gradient:
                self._gradient[lookup[k]] = fk_gradient[k]

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
