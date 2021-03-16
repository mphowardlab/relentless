"""
Algorithms
==========

An optimization algorithm seeks to determine the minima of a defined objective
function, subject to design constraints.

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

.. autoclass:: Optimizer
    :member-order: bysource
    :members: optimize,
        has_converged,
        abs_tol

.. autoclass:: SteepestDescent
    :member-order: bysource
    :members: optimize,
        step_size,
        max_iter

"""
import abc

import numpy as np

from .objective import ObjectiveFunction

class Optimizer(abc.ABC):
    """Abstract base class for optimization algorithm.

    A :class:`Optimizer` defines the optimization algorithm with specified parameters.

    Parameters
    ----------
    abs_tol : float or dict
        The absolute tolerance or tolerances (keyed on the :class:`~relentless.optimize.ObjectiveFunction`
        design variables).

    """
    def __init__(self, abs_tol):
        self.abs_tol = abs_tol

    @abc.abstractmethod
    def optimize(self, objective):
        """Minimize an objective function.

        The design variables of the objective function are adjusted until convergence
        (see :meth:`has_converged()`).

        Parameters
        ----------
        objective : :class:`~relentless.optimize.ObjectiveFunction`
            The objective function to be optimized.

        """
        pass

    def has_converged(self, result):
        """Check if the convergence criteria is satisfied.

        The absolute value of the gradient for each design variable of the objective
        function must be less than the absolute tolerance for that variable.

        Parameters
        ----------
        result : :class:`~relentless.optimize.ObjectiveFunctionResult`
            The computed value of the objective function.

        Returns
        -------
        bool
            ``True`` if the criteria is satisfied.

        Raises
        ------
        KeyError
            If the absolute tolerance value is not set for all design variables.
        """
        for x in result.design_variables:
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

    For an :class:`~relentless.optimize.ObjectiveFunction` :math:`f\left(\mathbf{x}\right)`,
    the steepest descent seeks to approach a minimum of the function.

    With an initial numerical guess for all the design variables :math:`\mathbf{x}`,
    the following iterative calculation is performed:

    .. math::

        \mathbf{x}_{n+1} = \mathbf{x}_{n}-\alpha\nabla f\left(\mathbf{x}\right)

    where :math:`\alpha` is defined as the step size hyperparameter.

    Parameters
    ----------
    abs_tol : float or dict
        The absolute tolerance or tolerances (keyed on the :class:`~relentless.optimize.ObjectiveFunction`
        design variables).
    step_size : float or :class:`LineSearch`
        The step size hyperparameter (:math:`\alpha`). The parameter is either
        defined numerically or is the :class:`LineSearch` object to be used
        to find the optimal step size.
    max_iter : int
        The maximum number of optimization iterations allowed.

    """
    def __init__(self, abs_tol, step_size, max_iter):
        super().__init__(abs_tol)
        self.step_size = step_size
        self.max_iter = max_iter

    def optimize(self, objective):
        """Perform the steepest descent optimization for the given objective function.

        Parameters
        ----------
        objective : :class:`~relentless.optimize.ObjectiveFunction`
            The objective function to be optimized.

        Returns
        -------
        bool or None
            ``True`` if converged, ``False`` if not converged, ``None`` if no
            design variables are specified for the objective function.
        """
        dvars = objective.design_variables()
        if len(dvars) == 0:
            return None

        res = objective.compute()
        try:
            step = self.step_size.find(objective=objective, result_0=res)
        except AttributeError:
            step = self.step_size

        iter_num = 0
        converged = False
        while not converged and iter_num < self.max_iter:
            for x in dvars:
                x.value -= step*res.gradient(x)
            converged = self.has_converged(res)
            res = objective.compute()
            iter_num += 1

        return converged

    @property
    def step_size(self):
        r"""float or :class:`LineSearch`: The step size hyperparameter (:math:`\alpha`)."""
        return self._step_size

    @step_size.setter
    def step_size(self, value):
        try:
            if value <= 0:
                raise ValueError('The step size must be positive if defined as a float.')
        except TypeError:
            if not isinstance(value, LineSearch):
                raise ValueError('The step size must be a float or an instance of LineSearch.')
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

class LineSearch:
    def __init__(self, abs_tol, max_step_size, max_iter):
        self.abs_tol = abs_tol
        self.max_step_size = max_step_size
        self.max_iter = max_iter

    def find(self, objective, result_0=None):
        dvars = objective.design_variables()
        if result_0 is None:
            result_0 = objective.compute()
        dvars_0 = result_0.design_variables

        # compute normalized search direction, adjust variables based on max step size
        d = {}
        d_norm = np.linalg.norm(np.asarray(list(result_0._gradient.todict().values())))
        for x in dvars:
            d[x] = (-result_0.gradient(x))/d_norm
            x.value += self.max_step_size*d[x]
        result = objective.compute()

        # compute initial and post-adjustment target values
        target_0 = 0
        target = 0
        for x in dvars:
            target_0 += (d[x]*-result_0.gradient(x))
            target += (d[x]*-result.gradient(x))

        # check if max step size is acceptable
        if target > 0 or np.abs(target) < self.abs_tol: #combine conditions as (target > -self.abs_tol) ?
            return self.max_step_size

        # iterate to minimize target
        else:
            rstep = np.array([0, self.max_step_size])
            rtarget = np.array([target_0, target])
            iter_num = 0
            while np.abs(target) > self.abs_tol and iter_num < self.max_iter:
                # linear interpolation for step size
                step_size = ((rstep[0]*rtarget[1] - rstep[1]*rtarget[0])
                            /(rtarget[1] - rtarget[0]))

                # adjust variables based on new step size, compute target
                for x in dvars:
                    x.value = dvars_0[x]
                    x.value += step_size*d[x]
                result = objective.compute()
                target = 0
                for x in dvars:
                    target += (d[x]*-result.gradient(x))

                # update search intervals
                if target > 0:
                    rstep[0] = step_size
                    rtarget[0] = target
                else:
                    rstep[1] = step_size
                    rtarget[1] = target

                iter_num += 1

            for x in dvars:
                x.value = dvars_0[x]
            return step_size

    @property
    def abs_tol(self):
        """float: The absolute tolerance for the target. Must be non-negative."""
        return self._abs_tol

    @abs_tol.setter
    def abs_tol(self, value):
        try:
            if value < 0:
                raise ValueError('The absolute tolerance must be non-negative.')
            else:
                self._abs_tol = value
        except TypeError:
            raise TypeError('The absolute tolerance must be a scalar float.')

    @property
    def max_step_size(self):
        r"""float: The maximum acceptable step size hyperparameter (:math:`\alpha`)."""
        return self._max_step_size

    @max_step_size.setter
    def max_step_size(self, value):
        if value <= 0:
            raise ValueError('The maximum step size must be positive.')
        self._max_step_size = value

    @property
    def max_iter(self):
        """int: The maximum number of line search iterations allowed."""
        return self._max_iter

    @max_iter.setter
    def max_iter(self, value):
        if not isinstance(value, int):
            raise TypeError('The maximum number of iterations must be an integer.')
        if value < 1:
            raise ValueError('The maximum number of iterations must be positive.')
        self._max_iter = value
