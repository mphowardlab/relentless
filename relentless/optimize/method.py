"""
Algorithms
==========

An optimization algorithm seeks to determine the minima of a defined objective
function, subject to design constraints.

The following optimization algorithms have been implemented:

.. autosummary::
    :nosignatures:

    SteepestDescent
    FixedStepDescent

.. rubric:: Developer notes

To implement your own optimization algorithm, create a class that derives from
:class:`Optimizer` and define the required properties and methods.

.. autosummary::
    :nosignatures:

    Optimizer
    LineSearch

.. autoclass:: Optimizer
    :member-order: bysource
    :members: optimize,
        has_converged,
        abs_tol

.. autoclass:: LineSearch
    :member-order: bysource
    :members: find,
        abs_tol,
        max_iter

.. autoclass:: SteepestDescent
    :member-order: bysource
    :members: descent_amount,
        optimize,
        max_iter,
        step_size,
        line_search

.. autoclass:: FixedStepDescent
    :member-order: bysource
    :members: descent_amount

"""
import abc

import numpy as np

from relentless import _collections
from .objective import ObjectiveFunction

class Optimizer(abc.ABC):
    """Abstract base class for optimization algorithm.

    A :class:`Optimizer` defines the optimization algorithm with specified parameters.

    Parameters
    ----------
    abs_tol : float or dict
        The absolute tolerance or tolerances (keyed on the :class:`~relentless.optimize.objective.ObjectiveFunction`
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
        objective : :class:`~relentless.optimize.objective.ObjectiveFunction`
            The objective function to be optimized.

        """
        pass

    def has_converged(self, result):
        """Check if the convergence criteria is satisfied.

        The absolute value of the gradient for each design variable of the objective
        function must be less than the absolute tolerance for that variable.

        Parameters
        ----------
        result : :class:`~relentless.optimize.objective.ObjectiveFunctionResult`
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
            grad = result.gradient[x]
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

class LineSearch:
    r"""Line search algorithm.

    For an :class:`~relentless.optimize.objective.ObjectiveFunction` :math:`f\left(\mathbf{x}\right)`,
    the line search algorithm seeks to take a single, optimally-sized step along
    a search interval, which must be a descent direction (a vector which points
    towards or crosses a local minimum of the objective function).

    A search interval :math:`\mathbf{d}` is specified, and normalized to :math:`\hat{\mathbf{d}}`:

    .. math::

        \mathbf{d} = \mathbf{x}_{end} - \mathbf{x}_{start}

        \hat{\mathbf{d}} = \frac{\mathbf{d}}{\lVert\mathbf{d}\rVert}

    Then, a "target" value :math:`t` is defined as:

    .. math::

        t = -\hat{\mathbf{d}} \cdot \nabla{f\left(\mathbf{x}\right)}

    Because :math:`\hat{\mathbf{d}}` is a descent direction, the target at the
    start of the search interval is always positive. If the target is positive
    (or within the tolerance) at the end of the search interval, then the maximum
    step size is acceptable and the algorithm steps to the end of the search
    interval. If the target is negative (outside of the tolerance) at the end of
    the search interval, then the algorithm iteratively computes a new step size
    by linear interpolation within the search interval until the target at the
    new location is minimized to within the tolerance.

    Note: This algorithm applies the
    `strong Wolfe condition on curvature <https://wikipedia.org/wiki/Wolfe_conditions#Strong_Wolfe_condition_on_curvature>`_.

    Parameters
    ----------
    abs_tol : float
        The absolute tolerance for the target.
    max_iter : int
        The maximum number of line search iterations allowed.

    """
    def __init__(self, abs_tol, max_iter):
        self.abs_tol = abs_tol
        self.max_iter = max_iter

    def find(self, objective, start, end):
        """Apply the line search algorithm to take the optimal step.

        Note that the objective function is kept at its initial state, and the
        function evaluted after taking the optimal step is returned separately.

        Parameters
        ----------
        objective : :class:`~relentless.optimize.objective.ObjectiveFunction`
            The objective function for which the line search is applied.
        start : :class:`~relentless.optimize.objective.ObjectiveFunctionResult`
            The objective function evaluated at the start of the search interval.
        end : :class:`~relentless.optimize.objective.ObjectiveFunctionResult`
            The objective function evaluated at the end of the search interval.

        Raises
        ------
        ValueError
            If the defined search interval is not a descent direction.

        Returns
        -------
        :class:`~relentless.optimize.objective.ObjectiveFunctionResult`
            The objective function evaluated at the new, "optimal" location.

        """
        ovars = {x: x.value for x in objective.design_variables()}

        # compute normalized search direction
        d = end.design_variables - start.design_variables
        max_step = d.norm()
        d /= max_step

        # compute start and end target values
        targets = np.array([-d.dot(start.gradient), -d.dot(end.gradient)])
        if targets[0] < 0:
            raise ValueError('The defined search interval must be a descent direction.')

        # check if max step size acceptable, else iterate to minimize target
        if targets[1] > 0 or np.abs(targets[1]) < self.abs_tol:
            result = end
        else:
            steps = np.array([0, max_step])
            iter_num = 0
            new_target = np.inf
            new_res = targets[1]
            while np.abs(new_target) >= self.abs_tol and iter_num < self.max_iter:
                # linear interpolation for step size
                new_step = (steps[0]*targets[1] - steps[1]*targets[0])/(targets[1] - targets[0])

                # adjust variables based on new step size, compute target
                for x in ovars:
                    x.value = start.design_variables[x] + new_step*d[x]
                new_res = objective.compute()
                new_target = -d.dot(new_res.gradient)

                # update search intervals
                if new_target > 0:
                    steps[0] = new_step
                    targets[0] = new_target
                else:
                    steps[1] = new_step
                    targets[1] = new_target

                iter_num += 1

            result = new_res

        for x in ovars:
            x.value = ovars[x]
        return result

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

class SteepestDescent(Optimizer):
    r"""Steepest descent algorithm.

    For an :class:`~relentless.optimize.objective.ObjectiveFunction` :math:`f\left(\mathbf{x}\right)`,
    the steepest descent algorithm seeks to approach a minimum of the function.

    With an initial numerical guess for all the design variables :math:`\mathbf{x}`,
    the following iterative calculation is performed:

    .. math::

        \mathbf{x}_{n+1} = \mathbf{x}_{n}-\alpha\nabla f\left(\mathbf{x}\right)

    where :math:`\alpha` is defined as the step size hyperparameter. The step size
    must be specified numerically, and a :class:`LineSearch` can be performed to find
    an optimal step size value if desired.

    Parameters
    ----------
    abs_tol : float or dict
        The absolute tolerance or tolerances (keyed on the :class:`~relentless.optimize.objective.ObjectiveFunction`
        design variables).
    step_size : dict or float
        The step size hyperparameter (:math:`\alpha`), as either a single value
        or keyed on the :class:`~relentless.optimize.objective.ObjectiveFunction`
        design variables).
    max_iter : int
        The maximum number of optimization iterations allowed.
    line_search : :class:`LineSearch`
        The line search object used to find the optimal step size, using the
        specified step size value as the "maximum" step size (defaults to ``None``).

    """
    def __init__(self, abs_tol, max_iter, step_size, line_search=None):
        super().__init__(abs_tol)
        self.max_iter = max_iter
        self.step_size = step_size
        self.line_search = line_search

    def descent_amount(self, gradient):
        r"""Calculate the descent amount for the optimization.

        The descent amount is:

        .. math::

            d = \alpha\lVert\nabla f\left(\mathbf{x}\right)\rVert

        Parameters
        ----------
        gradient : :class:`~relentless._collections.KeyedArray`
            The gradient of the objective function.

        Returns
        -------
        :class:`~relentless._collections.KeyedArray`
            The descent amount, keyed on the objective function design variables.

        """
        k = _collections.KeyedArray(keys=gradient.keys)
        try:
            k.update(self.step_size)
        except TypeError:
            for i in k:
                k[i] = self.step_size
        return k*gradient.norm()

    def optimize(self, objective):
        r"""Perform the steepest descent optimization for the given objective function.

        A single steepest descent update is performed as:

        .. math::

            \mathbf{x}_{n+1} = \mathbf{x}_{n}-d\frac{\nabla f\left(\mathbf{x}\right)}
                                                    {\lVert\nabla f\left(\mathbf{x}\right)\rVert}

        :math:`d` is the descent amount returned by :meth:`descent_amount()`.

        If specified, the line search is used as described in :class:`LineSearch`
        to place the objective function at a location such that the specified
        step size will reach a minimum.

        Parameters
        ----------
        objective : :class:`~relentless.optimize.objective.ObjectiveFunction`
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

        iter_num = 0
        cur_res = objective.compute()
        alpha = self.descent_amount(cur_res.gradient)
        update = alpha*cur_res.gradient/cur_res.gradient.norm()
        while not self.has_converged(cur_res) and iter_num < self.max_iter:
            #steepest descent update
            for x in dvars:
                x.value = cur_res.design_variables[x] - update[x]
            next_res = objective.compute()
            alpha = self.descent_amount(cur_res.gradient)
            update = alpha*cur_res.gradient/cur_res.gradient.norm()

            #if line search, attempt backtracking in interval
            if self.line_search is not None:
                line_res = self.line_search.find(objective=objective, start=cur_res, end=next_res)
                for x in dvars:
                    x.value = line_res.design_variables[x]
                next_res = line_res

            #recycle next result
            cur_res = next_res
            iter_num += 1

        return self.has_converged(cur_res)

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

    @property
    def step_size(self):
        r"""dict or float: The step size hyperparameter(s) (:math:`\alpha`).
        Must be positive."""
        return self._step_size

    @step_size.setter
    def step_size(self, value):
        try:
            step_size = dict(value)
            err = any([step <= 0  for step in value.values()])
        except TypeError:
            step_size = value
            err = value <= 0
        if err:
            raise ValueError('The step sizes must be positive.')
        self._step_size = value

    @property
    def line_search(self):
        """:class:`LineSearch`: The line search object used to optimize the step size."""
        return self._line_search

    @line_search.setter
    def line_search(self, value):
        if value is not None and not isinstance(value, LineSearch):
            raise TypeError('If defined, the line search parameter must be a LineSearch object.')
        self._line_search = value

class FixedStepDescent(SteepestDescent):
    r"""Fixed-step steepest descent algorithm.

    For an :class:`~relentless.optimize.objective.ObjectiveFunction` :math:`f\left(\mathbf{x}\right)`,
    the fixed-step steepest descent algorithm seeks to approach a minimum of the
    function.

    With an initial numerical guess for all the design variables :math:`\mathbf{x}`,
    the following iterative calculation is performed:

    .. math::

        \mathbf{x}_{n+1} = \mathbf{x}_{n}-\alpha\frac{\nabla f\left(\mathbf{x}\right)}
                                                     {\lVert\nabla f\left(\mathbf{x}\right)\rVert}

    where :math:`\alpha` is defined as the step size hyperparameter. The step size
    must be specified numerically, and a :class:`LineSearch` can be performed to find
    an optimal step size value if desired.

    Parameters
    ----------
    abs_tol : float or dict
        The absolute tolerance or tolerances (keyed on the :class:`~relentless.optimize.objective.ObjectiveFunction`
        design variables).
    step_size : dict or float
        The step size hyperparameter (:math:`\alpha`), as either a single value
        or keyed on the :class:`~relentless.optimize.objective.ObjectiveFunction`
        design variables).
    max_iter : int
        The maximum number of optimization iterations allowed.
    line_search : :class:`LineSearch`
        The line search object used to find the optimal step size, using the
        specified step size value as the "maximum" step size (defaults to ``None``).

    """
    def __init__(self, abs_tol, max_iter, step_size, line_search=None):
        super().__init__(abs_tol, max_iter, step_size, line_search)

    def descent_amount(self, gradient):
        r"""Calculate the descent amount for the optimization.

        The descent amount is:

        .. math::

            d = \alpha

        Parameters
        ----------
        gradient : :class:`~relentless._collections.KeyedArray`
            The gradient of the objective function.

        Returns
        -------
        :class:`~relentless._collections.KeyedArray`
            The descent amount, keyed on the objective function design variables.

        """
        k = _collections.KeyedArray(keys=gradient.keys)
        try:
            k.update(self.step_size)
        except TypeError:
            for i in k:
                k[i] = self.step_size
        return k
