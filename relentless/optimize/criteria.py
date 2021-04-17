import abc

import numpy as np

"""
    abs_tol : float or dict
        The absolute tolerance or tolerances (keyed on the :class:`~relentless.optimize.objective.ObjectiveFunction`
        design variables).
    rel_tol : float or dict
        The relative tolerance or tolerances (keyed on the :class:`~relentless.optimize.objective.ObjectiveFunction`
        design variables). (Defaults to ``None``).

"""
class ConvergenceTest(abc.ABC):
    @abc.abstractmethod
    def converged(self, result):
        pass

class AbsoluteTolerance(ConvergenceTest):
    def __init__(self, tolerance):
        self.tolerance = tolerance

    @property
    def tolerance(self):
        """float or dict: The absolute tolerance(s). Must be non-negative."""
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value):
        try:
            tol = dict(value)
            err = any([t < 0 for t in value.values()])
        except TypeError:
            tol = value
            err = value < 0
        if err:
            raise ValueError('Absolute tolerances must be non-negative.')
        else:
            self._tolerance = tol

    def _get_tol(self, var):
        try:
            t = self.tolerance[var]
        except TypeError:
            t = self.tolerance
        except KeyError:
            raise KeyError('An absolute tolerance is not set for design variable ' + str(var))
        return t

class RelativeTolerance(ConvergenceTest):
    def __init__(self, tolerance):
        self.tolerance = tolerance

    @property
    def tolerance(self):
        """float or dict: The relative tolerance(s). Must be between 0 and 1. Defaults to 0."""
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value):
        try:
            tol = dict(value)
            err = any([t < 0 or t > 1 for t in value.values()])
        except TypeError:
            tol = value
            err = value < 0 or value > 1
        except TypeError:
            tol = 0.
        if err:
            raise ValueError('Relative tolerances must be between 0 and 1.')
        else:
            self._tolerance = tol

    def _get_tol(self, var):
        try:
            t = self.tolerance[var]
        except TypeError:
            t = self.tolerance
        except KeyError:
            raise KeyError('A relative tolerance is not set for design variable ' + str(var))
        return t

class AbsoluteGradientTest(AbsoluteTolerance):
    def __init__(self, tolerance):
        super().__init__(tolerance)

    def converged(self, result):
        converged = True
        for x in result.design_variables:
            grad = result.gradient[x]
            tol = self._get_tol(x)
            if x.athigh() and -grad < -tol:
                converged = False
                break
            elif x.atlow() and -grad > tol:
                converged = False
                break
            elif x.isfree() and np.abs(grad) > tol:
                converged = False
                break

        return converged

class RelativeGradientTest(RelativeTolerance):
    def __init__(self, tolerance):
        super().__init__(tolerance)

    def converged(self, result):
        converged = True
        for x in result.design_variables:
            grad = result.gradient[x]
            tol =  self._get_tol(x)
            if x.athigh() and -grad < -tol:
                converged = False
                break
            elif x.atlow() and -grad > tol:
                converged = False
                break
            elif x.isfree() and np.abs(grad/x.value) > tol:
                converged = False
                break

        return converged

class ValueTest(AbsoluteTolerance):
    def __init__(self, tolerance):
        super().__init__(tolerance)

    def converged(self, result):
        converged = True
        for x in result.design_variables:
            tol = self._get_tol(x)
            if x.value > tol:
                converged = False
                break

        return converged

class LogicTest(ConvergenceTest):
    def __init__(self, *tests):
        if not all([isinstance(t, ConvergenceTest) for t in tests]):
            raise TypeError('All inputs to a LogicTest must be ConvergenceTests.')
        self.tests = tests

class AnyTest(LogicTest):
    def converged(self, result):
        return any([t.converged(result) for t in self.tests])

class AllTest(LogicTest):
    def converged(self, result):
        return all([t.converged(result) for t in self.tests])
