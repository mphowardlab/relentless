"""
Methods
=======

A :class:`Optimizer` defines an optimization algorithm that can be applied to
a defined :class:`ObjectiveFunction`.

The following algorithms have been implemented:

.. autosummary::
    :nosignatures:

    SteepestDescent

.. rubric:: Developer notes

To implement your own optimization algorithm, create a class that derives from
:class:`Optimizer` and define the required properties and methods.

.. autosummary::
    :nosignatures:

    Optimizer
    SteepestDescent

.. autoclass:: Optimizer
    :member-order: bysource
    :members: optimize
        has_converged
        abs_tol

.. autoclass:: SteepestDescent
    :member-order: bysource
    :members: optimize,
        step_size,
        max_iter

"""
import abc

from .objective import ObjectiveFunction

class Optimizer(abc.ABC):
    """Abstract base class for optimization algorithm.

    A :py:class:`Optimizer` defines the optimization algorithm with specified parameters.

    Parameters
    ----------
    abs_tol : float or dict
        The absolute tolerance (as a float) or tolerances (as a dict keyed on the
        :py:class:`ObjectiveFunction`'s design variables).

    """
    def __init__(self, abs_tol):
        self.abs_tol = abs_tol

    @abc.abstractmethod
    def optimize(self, objective):
        """Adjusts the given objective function until the convergence criteria
        (defined by :py:meth:`has_converged()`) is satisfied.

        Parameters
        ----------
        objective : :py:class:`ObjectiveFunction`
            The objective function to be optimized.

        """
        pass

    def has_converged(self, result):
        """Checks if the convergence criteria is satisfied.

        Checks if the absolute value of the gradient for each design variable of the
        objective function is less than the absolute tolerance for that variable.

        Parameters
        ----------
        result : :py:class:`ObjectiveFunctionResult`
            The computed value of the objective function.

        Returns
        -------
        bool
            ``True`` if the criteria is satisfied, ``False`` otherwise.

        Raises
        ------
        KeyError
            If the absolute tolerance value is not set for all design variables.
        """
        for x in result.design_variables():
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

class SteepestDescent(Optimizer):
    r"""Steepest descent algorithm.

    Parameters
    ----------
    abs_tol : float or dict
        The absolute tolerance (as a float) or tolerances (as a dict keyed on the
        :py:class:`ObjectiveFunction`'s design variables).
    step_size : float
        The step size hyperparameter for the optimization.
    max_iter : int
        The maximum number of optimization iterations allowed.

    """
    def __init__(self, abs_tol, step_size, max_iter):
        super().__init__(abs_tol)
        self.step_size = step_size
        self.max_iter = max_iter

    def optimize(self, objective):
        """Performs the steepest descent optimization for the given objective function.

        Parameters
        ----------
        objective : :py:class:`ObjectiveFunction`
            The objective function to be optimized.

        Returns
        -------
        None
            If no design variables are specified for the objective function.
        bool
            ``True`` if converged, ``False`` otherwise.
        """
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
        """float: The step size hyperparameter for the optimization."""
        return self._step_size

    @step_size.setter
    def step_size(self, value):
        if value <= 0:
            raise ValueError('The step size must be positive.')
        self._step_size = value

    @property
    def max_iter(self):
        """int: The maximum number of optimization iterations allowed."""
        return self._max_iter

    @max_iter.setter
    def max_iter(self, value):
        if not isinstance(value, int):
            raise TypeError('The maximum number of iterations must be an integer.')
        if value < 1:
            raise ValueError('The maximum number of iterations must be positive.')
        self._max_iter = value
