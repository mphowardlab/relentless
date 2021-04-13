import abc

import numpy as np

class ConvergenceTest(abc.ABC):
    """
        abs_tol : float or dict
            The absolute tolerance or tolerances (keyed on the :class:`~relentless.optimize.objective.ObjectiveFunction`
            design variables).
        rel_tol : float or dict
            The relative tolerance or tolerances (keyed on the :class:`~relentless.optimize.objective.ObjectiveFunction`
            design variables). (Defaults to ``None``).

    """
    def __init__(self, abs_tol, rel_tol=None, mode='or'):
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        if mode not in ('and','or'):
            raise ValueError('The tolerance mode must be specified as AND or OR.')
        self.mode = mode

    @abc.abstractmethod
    def converged(self, result):
        pass

    @property
    def abs_tol(self):
        """float or dict: The absolute tolerance(s). Must be non-negative."""
        return self._abs_tol

    @abs_tol.setter
    def abs_tol(self, value):
        try:
            abs_tol = dict(value)
            err = any([tol < 0 for tol in value.values()])
        except TypeError:
            abs_tol = value
            err = value < 0
        if err:
            raise ValueError('Absolute tolerances must be non-negative.')
        else:
            self._abs_tol = abs_tol

    @property
    def rel_tol(self):
        """float or dict: The relative tolerance(s). Must be between 0 and 1. Defaults to 0."""
        return self._rel_tol

    @rel_tol.setter
    def rel_tol(self, value):
        try:
            rel_tol = dict(value)
            err = any([tol <= 0 or tol >= 1 for tol in value.values()])
        except TypeError:
            rel_tol = value
            err = value <= 0 or value >= 1
        except TypeError:
            rel_tol = 0.
        if err:
            raise ValueError('Relative tolerances must be between 0 and 1.')
        else:
            self._rel_tol = rel_tol

    def _var_tol(self, var, tol):
        try:
            t = tol[x]
        except TypeError:
            t = tol
        except KeyError('A tolerance is not set for design variable {}.'.format(x))

class GradientTest(ConvergenceTest):
    def converged(self, result):
        converged = True
        for x in result.design_variables:
            grad = result.gradient[x]
            abs_tol = _var_tol(x, self.abs_tol)

            if x.is_upper() and -grad < -abs_tol:
                converged = False
                break
            elif x.is_lower() and -grad > abs_tol:
                converged = False
                break
            elif x.is_free():
                rel_tol = _var_tol(x, self.rel_tol)
                a = np.abs(grad) > abs_tol
                r = np.abs(grad/x.value) > rel_tol

                if self.mode == 'and':
                    if not a or not r:
                        converged = False
                        break
                elif self.mode == 'or':
                    if not a and not r:
                        converged = False
                        break

        return converged

class ValueTest(ConvergenceTest):
    def converged(self, result):
        converged = True
        for x in result.design_variables:
            abs_tol = _var_tol(x, self.abs_tol)
            if x.value > abs_tol:
                converged = False
                break

        return converged

class LogicTest(abc.ABC):
    def __init__(self, a, b):
        if not isinstance(a, ConvergenceTest) or not isinstance(b, ConvergenceTest):
            raise TypeError('Both inputs to a LogicTest must be ConvergenceTests.')
        self.a = a
        self.b = b

    @abc.abstractmethod
    def converged(self, result):
        pass

class OrTest(LogicTest):
    def converged(self, result):
        return self.a.converged(result) or self.b.converged(result)

class AndTest(LogicTest):
    def converged(self, result):
        return self.a.converged(result) and self.b.converged(result)
