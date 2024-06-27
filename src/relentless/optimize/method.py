import abc

import numpy

from relentless import data, math, mpi
from relentless.model import variable

from .criteria import ConvergenceTest, Tolerance


class Optimizer(abc.ABC):
    """Abstract base class for optimization algorithm.

    A :class:`Optimizer` defines the optimization algorithm with specified parameters.

    Parameters
    ----------
    stop : :class:`~relentless.optimize.criteria.ConvergenceTest`
        The convergence test used as the stopping criterion for the optimizer.

    """

    def __init__(self, stop):
        self.stop = stop

    @abc.abstractmethod
    def optimize(self, objective, variables, directory=None, overwrite=False):
        """Minimize an objective function.

        The design variables of the objective function are adjusted until convergence.

        Parameters
        ----------
        objective : :class:`~relentless.optimize.objective.ObjectiveFunction`
            The objective function to be optimized.
        variables: :class:`~relentless.variable.IndependentVariable` or tuple
            Design variable(s) to optimize.
        directory : str or :class:`~relentless.data.Directory`
            Directory for writing output during optimization. Default of ``None``
            requests no output is written.
        overwrite : bool
            If ``True``, overwrite the directory before beginning optimization.

        """
        pass

    @property
    def stop(self):
        """:class:`~relentless.optimize.criteria.ConvergenceTest`: The convergence
        test used as the stopping criterion for the optimizer."""
        return self._stop

    @stop.setter
    def stop(self, value):
        if not isinstance(value, ConvergenceTest):
            raise TypeError("The stopping criterion must be a ConvergenceTest.")
        self._stop = value

    def _setup_directory(self, directory, overwrite):
        directory = data.Directory.cast(directory, create=mpi.world.rank_is_root)
        mpi.world.barrier()
        if not directory.is_empty():
            if overwrite is True:
                if mpi.world.rank_is_root:
                    directory.clear_contents()
                mpi.world.barrier()
            else:
                raise OSError(
                    "Directory {} is not empty and overwrite is not True".format(
                        directory.path
                    )
                )
        return directory


class LineSearch:
    r"""Line search algorithm

    Given an :class:`~relentless.optimize.objective.ObjectiveFunction`
    :math:`f\left(\mathbf{x}\right)` and a search interval defined as
    :math:`\mathbf{d}=\mathbf{x}_{end}-\mathbf{x}_{start}`, the line search
    algorithm seeks an optimal step size :math:`0<\alpha<1` such
    that the following quantity is minimized:

    .. math::

        f\left(\mathbf{x}_{start}+\alpha\mathbf{d}\right)

    This is done by defining a scalar "target" value :math:`t` as:

    .. math::

        t = -\mathbf{d} \cdot \nabla{f\left(\mathbf{x}\right)}

    One measure of the optimal value of :math:`\alpha` is when :math:`t` decreases
    sufficiently relative to its value at the start of the interval:

    .. math::

        t(\alpha) < c\left\lvert t_{start}\right\rvert

    where :math:`c` is a defined relative tolerance value, and :math:`t_{start}`
    is the target value at the start of the search interval. This is the
    `strong Wolfe condition on curvature
    <https://wikipedia.org/wiki/Wolfe_conditions#Strong_Wolfe_condition_on_curvature>`_.

    Because :math:`\mathbf{d}` is a descent direction, the target at the
    start of the search interval is always positive. If the target is positive
    (or within the tolerance) at the end of the search interval, then the maximum
    step size is acceptable and the algorithm steps to the end of the search
    interval. If the target is negative (outside of the tolerance) at the end of
    the search interval, then the algorithm iteratively computes a new step size
    by linear interpolation within the search interval until the target at the
    new location is minimized to within the tolerance.

    If ``directory`` is specified,  one directory is created for each iteration
    of the line search, e.g., ``directory/0``.

    Parameters
    ----------
    tolerance : float
        The relative tolerance for the target (:math:`c`).
    max_iter : int
        The maximum number of line search iterations allowed.

    """

    def __init__(self, tolerance, max_iter):
        self.tolerance = tolerance
        self.max_iter = max_iter

    def find(self, objective, start, end, directory=None, scale=1.0):
        r"""Apply the line search algorithm to take the optimal step.

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
        directory : str or :class:`~relentless.data.Directory`
            Directory for writing output during search. Default of ``None``
            requests no output is written.
        scale : float or dict
            A scalar scaling parameter or scaling parameters
            (:math:\mathbf{X}`) keyed on one or more
            :class:`~relentless.optimize.objective.ObjectiveFunction` design
            variables (defaults to ``1.0``, so that the variables are unscaled).

        Returns
        -------
        :class:`~relentless.optimize.objective.ObjectiveFunctionResult`
            The objective function evaluated at the new, "optimal" location.

        """
        if directory is not None:
            directory = data.Directory.cast(directory, create=mpi.world.rank_is_root)
            mpi.world.barrier()
        ovars = {x: x.value for x in start.variables}

        scale_array = math.KeyedArray(keys=start.variables)
        for x in start.variables:
            if numpy.isscalar(scale):
                scale_array[x] = scale
            else:
                try:
                    scale_array[x] = scale[x]
                except KeyError:
                    scale_array[x] = 1.0

        # compute search direction
        d = (end.variables - start.variables) / scale_array
        if d.norm() == 0:
            raise ValueError(
                "The start and end of the search interval must be different."
            )

        # compute start and end target values
        targets = numpy.array(
            [-d.dot(start.gradient * scale_array), -d.dot(end.gradient * scale_array)]
        )
        if targets[0] < 0:
            raise ValueError("The defined search interval must be a descent direction.")

        # compute tolerance
        tol = Tolerance(absolute=self.tolerance * numpy.abs(targets[0]), relative=0)

        # check if max step size acceptable, else iterate to minimize target
        if targets[1] > 0 or tol.isclose(targets[1], 0):
            result = end
        else:
            steps = numpy.array([0.0, 1.0])
            iter_num = 0
            new_target = numpy.inf
            new_res = end
            while not tol.isclose(new_target, 0) and iter_num < self.max_iter:
                # linear interpolation for step size
                new_step = (steps[0] * targets[1] - steps[1] * targets[0]) / (
                    targets[1] - targets[0]
                )

                # adjust variables based on new step size, compute new target
                for x in start.variables:
                    x.value = start.variables[x] + scale_array[x] * new_step * d[x]
                if directory is not None:
                    new_dir = directory.directory(
                        str(iter_num), create=mpi.world.rank_is_root
                    )
                    mpi.world.barrier()
                else:
                    new_dir = None
                new_res = objective.compute(start.variables, new_dir)
                new_target = -d.dot(new_res.gradient * scale_array)

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
    def tolerance(self):
        """float: The relative tolerance for the target. Must be between 0 and 1."""
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value):
        if value < 0 or value > 1:
            raise ValueError("The relative tolerance must be between 0 and 1.")
        self._tolerance = value

    @property
    def max_iter(self):
        """int: The maximum number of line search iterations allowed."""
        return self._max_iter

    @max_iter.setter
    def max_iter(self, value):
        if not isinstance(value, int):
            raise TypeError("The maximum number of iterations must be an integer.")
        if value < 1:
            raise ValueError("The maximum number of iterations must be positive.")
        self._max_iter = value


class SteepestDescent(Optimizer):
    r"""Steepest descent algorithm.

    For an :class:`~relentless.optimize.objective.ObjectiveFunction`
    :math:`f\left(\mathbf{x}\right)`, the steepest descent algorithm seeks to
    approach a minimum of the function.

    The optimization is performed using scaled variables :math:`\mathbf{y}`.
    Define :math:`\mathbf{X}` as the scaling parameters for each variable such
    that :math:`y_i=x_i/X_i`. (A variable can be left unscaled by setting
    :math:`X_i=1`).

    Define :math:`\alpha` as the descent step size hyperparameter. A :class:`LineSearch`
    can optionally be performed to optimize the value of :math:`\alpha` between
    :math:`0` and the input value. The function is iteratively minimized by taking
    successive steps down the gradient of the function. If the scaled variables
    are :math:`\mathbf{y}_n` at iteration :math:`n`, the next value of the variables is:

    .. math::

        \mathbf{y}_{n+1} = \mathbf{y}_n-\alpha\nabla f\left(\mathbf{y}_n\right)

    The gradient of the function with respect to the scaled variables is:

    .. math::

        \nabla f\left(\mathbf{y}\right) =
            \left[X_1 \frac{\partial f}{\partial x_1},
            \cdots,
            X_n \frac{\partial f}{\partial x_n}\right]

    Note that this optimization procedure is equivalent to:

    .. math::

        \left(x_i\right)_{n+1} =
            \left(x_i\right)_n-\alpha{X_i}^2 \frac{\partial f}{\partial x_i}

    for each unscaled design variable :math:`x_i`.

    Parameters
    ----------
    stop : :class:`~relentless.optimize.criteria.ConvergenceTest`
        The convergence test used as the stopping criterion for the optimizer.
        Note that the result being tested will have *unscaled* variables and gradient.
    max_iter : int
        The maximum number of optimization iterations allowed.
    step_size : float
        The step size hyperparameter (:math:`\alpha`).
    scale : float or dict
        A scalar scaling parameter or scaling parameters (:math:`\mathbf{X}`)
        keyed on one or more :class:`~relentless.optimize.objective.ObjectiveFunction`
        design variables (defaults to ``1.0``, so that the variables are unscaled).
    line_search : :class:`LineSearch`
        The line search object used to find the optimal step size, using the
        specified step size value as the "maximum" step size (defaults to ``None``).

    """

    def __init__(self, stop, max_iter, step_size, scale=1.0, line_search=None):
        super().__init__(stop)
        self.max_iter = max_iter
        self.step_size = step_size
        self.scale = scale
        self.line_search = line_search

    def descent_amount(self, gradient):
        r"""Calculate the descent amount for the optimization.

        The amount that each update descends down the scaled gradient is a
        constant :math:`\alpha`.

        Parameters
        ----------
        gradient : :class:`~relentless.math.KeyedArray`
            The scaled gradient of the objective function.

        Returns
        -------
        :class:`~relentless.math.KeyedArray`
            The descent amount, keyed on the objective function design variables.

        """
        k = math.KeyedArray(keys=gradient.keys())
        for i in k:
            k[i] = self.step_size
        return k

    def optimize(self, objective, variables, directory=None, overwrite=False):
        r"""Perform the steepest descent optimization for the given objective function.

        If specified, a :class:`LineSearch` is performed to choose an optimal step size.

        If ``directory`` is specified and ``overwrite`` is ``True``, ``directory``
        will be cleared before the optimization begins. The output will be saved
        into a directory created for each iteration of the optimization, e.g.,
        ``directory/0``. To advance to the next iteration of the optimization
        (e.g., from iteration 0 to iteration 1), a directory ``directory/0/.next``
        is created at iteration 0 to hold the proposed result at iteration 1. If
        :attr:`line_search` is `None`, its contents are immediately moved to
        ``directory/1`` (leaving ``directory/0/.next``) empty. If :attr:`line_search`
        is not `None`, ``directory/0/.line`` will be created for :meth:`LineSearch.find`
        to use; the final result of the line search will be moved to ``directory/1``.

        Parameters
        ----------
        objective : :class:`~relentless.optimize.objective.ObjectiveFunction`
            The objective function to be optimized.
        variables: :class:`~relentless.variable.IndependentVariable` or tuple
            Design variable(s) to optimize.
        directory : str or :class:`~relentless.data.Directory`
            Directory for writing output during optimization. Default of `None`
            requests no output is written.
        overwrite : bool
            If ``True``, overwrite the directory before beginning optimization.

        Returns
        -------
        bool or None
            ``True`` if converged, ``False`` if not converged, ``None`` if no
            design variables are specified for the objective function.

        Raises
        ------
        OSError
            If ``directory`` is not empty and overwrite is ``False``.

        """
        variables = variable.graph.check_variables_and_types(
            variables, variable.IndependentVariable
        )
        if len(variables) == 0:
            return None

        if directory is not None:
            directory = self._setup_directory(directory, overwrite)

        # fix scaling parameters
        scale = math.KeyedArray(keys=variables)
        for x in variables:
            if numpy.isscalar(self.scale):
                scale[x] = self.scale
            else:
                try:
                    scale[x] = self.scale[x]
                except KeyError:
                    scale[x] = 1.0

        iter_num = 0
        if directory is not None:
            cur_dir = directory.directory(str(iter_num), create=mpi.world.rank_is_root)
            mpi.world.barrier()
        else:
            cur_dir = None
        cur_res = objective.compute(variables, cur_dir)
        while not self.stop.converged(cur_res) and iter_num < self.max_iter:
            grad_y = scale * cur_res.gradient
            update = scale * self.descent_amount(grad_y) * grad_y

            # steepest descent update
            for x in variables:
                x.value = cur_res.variables[x] - update[x]
            if cur_dir is not None:
                next_dir = cur_dir.directory(".next", create=mpi.world.rank_is_root)
                mpi.world.barrier()
            else:
                next_dir = None
            next_res = objective.compute(variables, next_dir)

            # if line search, attempt backtracking in interval
            if self.line_search is not None:
                if cur_dir is not None:
                    line_dir = cur_dir.directory(".line", create=mpi.world.rank_is_root)
                    mpi.world.barrier()
                else:
                    line_dir = None
                line_res = self.line_search.find(
                    objective=objective,
                    start=cur_res,
                    end=next_res,
                    directory=line_dir,
                    scale=self.scale,
                )

                if line_res is not next_res:
                    for x in variables:
                        x.value = line_res.variables[x]
                    next_res = line_res

            # move the contents of the "next" result to the new "current" result
            if directory is not None:
                cur_dir = directory.directory(
                    str(iter_num + 1), create=mpi.world.rank_is_root
                )
                mpi.world.barrier()
            else:
                cur_dir = None
            if next_res.directory is not None:
                mpi.world.barrier()
                if mpi.world.rank_is_root:
                    next_res.directory.move_contents(cur_dir)
                mpi.world.barrier()

            # recycle next result, updating directory to new location
            cur_res = next_res
            cur_res.directory = cur_dir
            iter_num += 1

        return self.stop.converged(cur_res)

    @property
    def max_iter(self):
        """int: The maximum number of optimization iterations allowed."""
        return self._max_iter

    @max_iter.setter
    def max_iter(self, value):
        if not isinstance(value, int):
            raise TypeError("The maximum number of iterations must be an integer.")
        if value < 1:
            raise ValueError("The maximum number of iterations must be positive.")
        self._max_iter = value

    @property
    def step_size(self):
        r"""float: The step size hyperparameter (:math:`\alpha`). Must be positive."""
        return self._step_size

    @step_size.setter
    def step_size(self, value):
        if value <= 0:
            raise ValueError("The step size must be positive.")
        self._step_size = value

    @property
    def scale(self):
        r"""float or dict: Scaling parameter.

        A scalar scaling parameter or scaling parameters (:math:`\mathbf{X}`)
        keyed on one or more :class:`~relentless.optimize.objective.ObjectiveFunction`
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
            raise ValueError("The scaling parameters must be positive.")
        self._scale = scale

    @property
    def line_search(self):
        """:class:`LineSearch`: The line search used to optimize the step size."""
        return self._line_search

    @line_search.setter
    def line_search(self, value):
        if value is not None and not isinstance(value, LineSearch):
            raise TypeError(
                "If defined, the line search parameter must be a LineSearch object."
            )
        self._line_search = value


class FixedStepDescent(SteepestDescent):
    r"""Fixed-step steepest descent algorithm.

    This is a modification of :class:`SteepestDescent` in which the function is
    iteratively minimized by taking successive steps of fixed magnitude down
    the normalized gradient of the function.

    If the scaled variables are :math:`\mathbf{y}_n` at iteration :math:`n`, the
    next value of the variables is:

    .. math::

        \mathbf{y}_{n+1} = \mathbf{y}_n
            -\frac{\alpha}{\left\lVert\nabla f\left(\mathbf{y}_n\right)\right\rVert}
            \nabla f\left(\mathbf{y}_n\right)

    Parameters
    ----------
    stop : :class:`~relentless.optimize.criteria.ConvergenceTest`
        The convergence test used as the stopping criterion for the optimizer.
        Note that the result being tested will have *unscaled* variables and gradient.
    max_iter : int
        The maximum number of optimization iterations allowed.
    step_size : float
        The step size hyperparameter (:math:`\alpha`).
    scale : float or dict
        A scalar scaling parameter or scaling parameters (:math:`\mathbf{X}`)
        keyed on one or more :class:`~relentless.optimize.objective.ObjectiveFunction`
        design variables (defaults to ``1.0``, so that the variables are unscaled).
    line_search : :class:`LineSearch`
        The line search object used to find the optimal step size, using the
        specified step size value as the "maximum" step size (defaults to ``None``).

    """

    def __init__(self, stop, max_iter, step_size, scale=1.0, line_search=None):
        super().__init__(stop, max_iter, step_size, scale, line_search)

    def descent_amount(self, gradient):
        r"""Calculate the descent amount for the optimization.

        The amount that each update descends down the scaled gradient is:

        .. math::

            \frac{\alpha}{\left\lVert\nabla y\right\rVert}

        which makes a step of constant magnitude.

        Parameters
        ----------
        gradient : :class:`~relentless.math.KeyedArray`
            The scaled gradient of the objective function.

        Returns
        -------
        :class:`~relentless.math.KeyedArray`
            The descent amount, keyed on the objective function design variables.

        """
        k = math.KeyedArray(keys=gradient.keys())
        for i in k:
            k[i] = self.step_size
        return k / gradient.norm()
