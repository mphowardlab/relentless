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
        scale_grad,
        optimize,
        max_iter,
        step_size,
        scale,
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

    Given an :class:`~relentless.optimize.objective.ObjectiveFunction` :math:`f\left(\mathbf{x}\right)`
    and a search interval defined by :math:`\mathbf{x}_{start}` and :math:`\mathbf{x}_{end}`,
    the line search algorithm seeks an optimal step size :math:`0<\alpha<1` such
    that :math:`f\left(\mathbf{x}_{start}+\alpha d\right)` is minimized.

    :math:`\mathbf{d}` is the search interval, which can be normalized to :math:`\mathbf{\hat{d}}` as:

    .. math::

        \mathbf{d} = \mathbf{x}_{end} - \mathbf{x}_{start}

        \mathbf{\hat{d}} = \frac{\mathbf{d}}{\lVert\mathbf{d}\rVert}

    Then, a scalar "target" value :math:`t` is defined as:

    .. math::

        t = -\mathbf{\hat{d}} \cdot \nabla{f\left(\mathbf{x}\right)}

    Because :math:`\mathbf{\hat{d}}` is a descent direction, the target at the
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
            If the start and the end of the search interval are identical.
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
        if max_step == 0:
            raise ValueError('The start and end of the search interval must be different.')
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

                # adjust variables based on new step size, compute new target
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

    :math:`\mathbf{X}=\left[X_1,\ldots,X_n\right]` is defined as the scaling
    parameters for the gradient such that :math:`y_i=\frac{x_i}{X_i}`. By
    default, the variables are left unscaled (:math:`X_i=1`).

    With scaling, the gradient of the function becomes:

    .. math::

        \nabla f\left(\mathbf{y}\right) = \left[X_1 \frac{\partial f}{\partial x_1},
                                                      \cdots,
                                                X_n \frac{\partial f}{\partial x_n}\right]

    :math:`\alpha` is defined as the step size hyperparameter. The step size must
    be specified numerically, and a :class:`LineSearch` can be performed to find
    an optimal step size value if desired.

    Then, with an initial numerical guess for all the design variables :math:`\mathbf{x}`,
    the following iterative calculation is performed:

    .. math::

        \mathbf{y}_{n+1} = \mathbf{y}_{n}-\alpha\nabla f\left(\mathbf{y}\right)

    such that each term is equivalent to:

    .. math::

        \left(x_i\right)_{n+1} = \left(x_i\right)_{n}-\alpha{X_i}^2 \frac{\partial f}{\partial x_i}

    Parameters
    ----------
    abs_tol : float or dict
        The absolute tolerance or tolerances keyed on the
        :class:`~relentless.optimize.objective.ObjectiveFunction` design variables.
        The tolerance is defined on the `scaled gradient`.
    max_iter : int
        The maximum number of optimization iterations allowed.
    step_size : float
        The step size hyperparameter (:math:`\alpha`).
    scale : float or dict
        A scalar scaling parameter or scaling parameters (:math:`\mathbf{X}`)
        keyed on one or more `~relentless.optimize.objective.ObjectiveFunction`
        design variables (defaults to ``1.0``, so that the variables are unscaled).
    line_search : :class:`LineSearch`
        The line search object used to find the optimal step size, using the
        specified step size value as the "maximum" step size (defaults to ``None``).

    """
    def __init__(self, abs_tol, max_iter, step_size, scale=1.0, line_search=None):
        super().__init__(abs_tol)
        self.max_iter = max_iter
        self.step_size = step_size
        self.scale = scale
        self.line_search = line_search

    def descent_amount(self, gradient):
        r"""Calculate the descent amount for the optimization.

        The amount that each update descends down the scaled gradient is:

        .. math::

            \alpha

        Parameters
        ----------
        gradient : :class:`~relentless._collections.KeyedArray`
            The scaled gradient of the objective function.

        Returns
        -------
        :class:`~relentless._collections.KeyedArray`
            The descent amount, keyed on the objective function design variables.

        """
        k = _collections.KeyedArray(keys=gradient.keys)
        for i in k:
            k[i] = self.step_size
        return k

    def scale_grad(self, gradient, scale):
        r"""Computes the scaled gradient of the objective function.

        The scaled gradient is defined as:

        .. math::

            \nabla f\left(\mathbf{y}\right) = \left[X_1 \frac{\partial f}{\partial x_1},
                                                          \cdots,
                                                    X_n \frac{\partial f}{\partial x_n}\right]

        Parameters
        ----------
        gradient : `~relentless._collections.KeyedArray`
            A gradient vector, keyed on one or more design variables.
        scale : float or dict
            A scalar scaling parameter or scaling parameters keyed on one or
            more design variables.

        Returns
        -------
        `~relentless._collections.KeyedArray`
            The scaled gradient.

        Raises
        ------
        KeyError
            If the scaling parameters are defined on a variable which is not in
            the objective function.

        """
        grad_sc = gradient
        if np.isscalar(scale):
            grad_sc *= scale
        else:
            for x in scale:
                try:
                    grad_sc[x] *= scale[x]
                except KeyError:
                    raise KeyError('''The scaling parameters cannot be defined
                                      on a variable which is not in the objective
                                      function.''')
        return grad_sc

    def optimize(self, objective):
        r"""Perform the steepest descent optimization for the given objective function.

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

        Raises
        ------
        ValueError
            If the line search used has a tolerance greater than the tolerance
            for the steepest descent.

        """
        dvars = objective.design_variables()
        if len(dvars) == 0:
            return None

        iter_num = 0
        cur_res = objective.compute()
        while not self.has_converged(cur_res) and iter_num < self.max_iter:
            grad_y = self.scale_grad(cur_res.gradient, self.scale)
            update = self.descent_amount(grad_y)*self.scale_grad(grad_y, self.scale)

            #steepest descent update
            for x in dvars:
                x.value = cur_res.design_variables[x] - update[x]
            next_res = objective.compute()

            #if line search, attempt backtracking in interval
            if self.line_search is not None:
                try:
                    err = self.line_search.abs_tol > self.abs_tol
                except TypeError:
                    err = any([self.line_search.abs_tol > t for t in self.abs_tol.values()])
                if err:
                    raise ValueError('''The line search object must have a tolerance less
                                        than or equal to the steepest descent object.''')

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
        r"""float: The step size hyperparameter (:math:`\alpha`). Must be positive."""
        return self._step_size

    @step_size.setter
    def step_size(self, value):
        if value <= 0:
            raise ValueError('The step size must be positive.')
        self._step_size = value

    @property
    def scale(self):
        r"""float or dict: A scalar scaling parameter or scaling parameters (:math:`\mathbf{X}`)
        keyed on one or more `~relentless.optimize.objective.ObjectiveFunction`
        design variables. Must be positive."""
        return self._scale

    @scale.setter
    def scale(self, value):
        try:
            scale = dict(value)
            err = any([s <= 0 for s in value.values()])
        except TypeError:
            scale = value
            err = value <= 0
        if err:
            raise ValueError('The scaling parameters must be positive.')
        self._scale = scale

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

    :math:`\mathbf{X}=\left[X_1,\ldots,X_n\right]` is defined as the scaling
    parameters for the gradient such that :math:`y_i=\frac{x_i}{X_i}`. By
    default, the variables are left unscaled (:math:`X_i=1`).

    With scaling, the gradient of the function becomes:

    .. math::

        \nabla f\left(\mathbf{y}\right) = \left[X_1 \frac{\partial f}{\partial x_1},
                                                      \cdots,
                                                X_n \frac{\partial f}{\partial x_n}\right]

    :math:`\alpha` is defined as the step size hyperparameter. The step size must
    be specified numerically, and a :class:`LineSearch` can be performed to find
    an optimal step size value if desired.

    Then, with an initial numerical guess for all the design variables :math:`\mathbf{x}`,
    the following iterative calculation is performed:

    .. math::

        \mathbf{y}_{n+1} = \mathbf{y}_{n}-\alpha\frac{\nabla f\left(\mathbf{y}\right)}
                                                     {\lVert\nabla f\left(\mathbf{y}\right)\rVert}

    such that each term is equivalent to:

    .. math::

        \left(x_i\right)_{n+1} = \left(x_i\right)_{n}-\frac{\alpha{X_i}^2 \frac{\partial f}{\partial x_i}}
                                                           {\sqrt{\sum\limits_{i=1}^{n} \left({X_i}^2 \frac{\partial f}{\partial x_i}\right)^2}}

    Parameters
    ----------
    abs_tol : float or dict
        The absolute tolerance or tolerances keyed on the
        :class:`~relentless.optimize.objective.ObjectiveFunction` design variables.
        The tolerance is defined on the `scaled gradient`.
    max_iter : int
        The maximum number of optimization iterations allowed.
    step_size : float
        The step size hyperparameter (:math:`\alpha`).
    scale : float or dict
        A scalar scaling parameter or scaling parameters (:math:`\mathbf{X}`)
        keyed on one or more `~relentless.optimize.objective.ObjectiveFunction`
        design variables (defaults to ``1.0``, so that the variables are unscaled).
    line_search : :class:`LineSearch`
        The line search object used to find the optimal step size, using the
        specified step size value as the "maximum" step size (defaults to ``None``).

    """
    def __init__(self, abs_tol, max_iter, step_size, scale=1.0, line_search=None):
        super().__init__(abs_tol, max_iter, step_size, scale, line_search)

    def descent_amount(self, gradient):
        r"""Calculate the descent amount for the optimization.

        The amount that each update descends down the scaled gradient is:

        .. math::

            \frac{\alpha}{\lVert\nabla y\rVert}

        Parameters
        ----------
        gradient : :class:`~relentless._collections.KeyedArray`
            The scaled gradient of the objective function.

        Returns
        -------
        :class:`~relentless._collections.KeyedArray`
            The descent amount, keyed on the objective function design variables.

        """
        k = _collections.KeyedArray(keys=gradient.keys)
        for i in k:
            k[i] = self.step_size
        return k/gradient.norm()
