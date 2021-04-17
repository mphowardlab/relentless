"""
Convergence Criteria
====================

A convergence test determines if an objective function has converged to the
desired minimum, subject to design constraints.

The following convergence tests have been implemented:

.. autosummary::
    :nosignatures:

    AbsoluteGradientTest
    AllTest
    AnyTest
    RelativeGradientTest
    ValueTest

.. rubric:: Developer notes

To implement your own convergence test, create a class that derives from any of
the four abstract base classes below, and define the required properties and methods.

.. autosummary::
    :nosignatures:

    AbsoluteTolerance
    ConvergenceTest
    LogicTest
    RelativeTolerance

.. autoclass:: ConvergenceTest
    :member-order: bysource
    :members: converged

.. autoclass:: AbsoluteTolerance
    :member-order: bysource
    :members: tolerance

.. autoclass:: AbsoluteGradientTest
    :member-order: bysource
    :members: converged

.. autoclass:: RelativeTolerance
    :member-order: bysource
    :members: tolerance

.. autoclass:: RelativeGradientTest
    :member-order: bysource
    :members: converged

.. autoclass:: ValueTest
    :member-order: bysource
    :members: converged

.. autoclass:: LogicTest
    :member-order: bysource
    :members: converged

.. autoclass:: AllTest
    :member-order: bysource
    :members: converged

.. autoclass:: AnyTest
    :member-order: bysource
    :members: converged

"""
import abc

import numpy as np

class ConvergenceTest(abc.ABC):
    r"""Abstract base class for optimization convergence tests.

    A :class:`ConvergenceTest` defines a test to determine if an
    :class:`~relentless.optimize.objective.ObjectiveFunction` :math:`f\left(\mathbf{x}\right)`,
    defined on a set of design variables :math:`\mathbf{x}=\left[x_1,\cdots,x_n\right]`,
    has converged to a desired point.

    """
    @abc.abstractmethod
    def converged(self, result):
        """Check if the function is converged.

        Parameters
        ----------
        result : :class:`~relentless.optimize.objective.ObjectiveFunctionResult`
            The location of the function at which to check for convergence.

        """
        pass

class AbsoluteTolerance(ConvergenceTest):
    """Abstract base class defining for absolute tolerance.

    The tolerance can be any non-negative numerical value.

    Parameters
    ----------
    tolerance : float or dict
        The absolute tolerance or tolerances (keyed on the :class:`~relentless.optimize.objective.ObjectiveFunction`
        design variables).

"""
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
        #helper method for standardizing between scalar/dict tolerance
        try:
            t = self.tolerance[var]
        except TypeError:
            t = self.tolerance
        except KeyError:
            raise KeyError('An absolute tolerance is not set for design variable ' + str(var))
        return t

class RelativeTolerance(ConvergenceTest):
    """Abstract base class defining for relative tolerance.

    The tolerance can be any non-negative numerical value between ``0`` and ``1``.

    Parameters
    ----------
    tolerance : float or dict
        The relative tolerance or tolerances (keyed on the :class:`~relentless.optimize.objective.ObjectiveFunction`
        design variables).

"""
    def __init__(self, tolerance):
        self.tolerance = tolerance

    @property
    def tolerance(self):
        """float or dict: The relative tolerance(s). Must be between ``0`` and ``1``."""
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value):
        try:
            tol = dict(value)
            err = any([t < 0 or t > 1 for t in value.values()])
        except TypeError:
            tol = value
            err = value < 0 or value > 1
        if err:
            raise ValueError('Relative tolerances must be between 0 and 1.')
        else:
            self._tolerance = tol

    def _get_tol(self, var):
        #helper method for standardizing between scalar/dict tolerance
        try:
            t = self.tolerance[var]
        except TypeError:
            t = self.tolerance
        except KeyError:
            raise KeyError('A relative tolerance is not set for design variable ' + str(var))
        return t

class AbsoluteGradientTest(AbsoluteTolerance):
    r"""Gradient test for convergence using absolute tolerance.

    The absolute tolerance, :math:`t`, can be any non-negative numerical value.

    The function is determined to be converged if any of the following conditions
    are satisfied for each of the design variables :math:`x_i`:

    .. math::

        -\frac{\partial f}{\partial x_i^{max}} > -t

        -\frac{\partial f}{\partial x_i^{min}} < t

        \left\lvert\frac{\partial f}{\partial x_i^{free}}\right\rvert < t

    :math:`x_i^{max}` refers to a design variable which is at the maximum value
    in its constrained domain, i.e. at :class:`~relentless.variable.DesignVariable.State` ``HIGH``.
    :math:`x_i^{min}` refers to a design variable which is at the minimum value
    in its constrained domain, i.e. at :class:`~relentless.variable.DesignVariable.State` ``LOW``.
    :math:`x_i^{free}` refers to a design variable which is not at the minimum
    or maximum value of its constrained domain, i.e. at :class:`~relentless.variable.DesignVariable.State` ``FREE``.

    Parameters
    ----------
    tolerance : float or dict
        The absolute tolerance or tolerances (keyed on the :class:`~relentless.optimize.objective.ObjectiveFunction`
        design variables).

"""
    def __init__(self, tolerance):
        super().__init__(tolerance)

    def converged(self, result):
        """Check if the function is converged using the absolute gradient test.

        Parameters
        ----------
        result : :class:`~relentless.optimize.objective.ObjectiveFunctionResult`
            The location of the function at which to check for convergence.

        Returns
        -------
        bool
            ``True`` if the function is converged.

        """
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
    r"""Gradient test for convergence using relative tolerance.

    The relative tolerance, :math:`t`, can be any non-negative numerical value
    between ``0`` and ``1``.

    The function is determined to be converged if any of the following conditions
    are satisfied for each of the design variables :math:`x_i`:

    .. math::

        -\frac{\partial f}{\partial x_i^{max}} > -t

        -\frac{\partial f}{\partial x_i^{min}} < t

        \left\lvert\frac{1}{x_i^{free}}\frac{\partial f}{\partial x_i^{free}}\right\rvert < t

    :math:`x_i^{max}` refers to a design variable which is at the maximum value
    in its constrained domain, i.e. at :class:`~relentless.variable.DesignVariable.State` ``HIGH``.
    :math:`x_i^{min}` refers to a design variable which is at the minimum value
    in its constrained domain, i.e. at :class:`~relentless.variable.DesignVariable.State` ``LOW``.
    :math:`x_i^{free}` refers to a design variable which is not at the minimum
    or maximum value of its constrained domain, i.e. at :class:`~relentless.variable.DesignVariable.State` ``FREE``.

    Parameters
    ----------
    tolerance : float or dict
        The relative tolerance or tolerances (keyed on the :class:`~relentless.optimize.objective.ObjectiveFunction`
        design variables).

"""
    def __init__(self, tolerance):
        super().__init__(tolerance)

    def converged(self, result):
        """Check if the function is converged using the relative gradient test.

        Parameters
        ----------
        result : :class:`~relentless.optimize.objective.ObjectiveFunctionResult`
            The location of the function at which to check for convergence.

        Returns
        -------
        bool
            ``True`` if the function is converged.

        """
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
    r"""Value test for convergence.

    The absolute "tolerance", :math:`t`, can be any non-negative numerical value.

    The function is determined to be converged if :math:`x_i<t` for each of the
    design variables :math:`x_i`.

    Parameters
    ----------
    tolerance : float or dict
        The absolute tolerance or tolerances (keyed on the :class:`~relentless.optimize.objective.ObjectiveFunction`
        design variables).

"""
    def __init__(self, tolerance):
        super().__init__(tolerance)

    def converged(self, result):
        """Check if the function is converged using the value test.

        Parameters
        ----------
        result : :class:`~relentless.optimize.objective.ObjectiveFunctionResult`
            The location of the function at which to check for convergence.

        Returns
        -------
        bool
            ``True`` if the function is converged.

        """
        converged = True
        for x in result.design_variables:
            tol = self._get_tol(x)
            if x.value > tol:
                converged = False
                break

        return converged

class LogicTest(ConvergenceTest):
    """Abstract base class for logical convergence tests.

    Parameters
    ----------
    tests : args
        The :class:`ConvergenceTest`\s to be used.

    Raises
    ------
    TypeError
        If all inputs are not :class:`ConvergenceTest`\s.

    """
    def __init__(self, *tests):
        if not all([isinstance(t, ConvergenceTest) for t in tests]):
            raise TypeError('All inputs to a LogicTest must be ConvergenceTests.')
        self.tests = tests

class AnyTest(LogicTest):
    """Logic test if any specified test returns convergence.

    Check if the function is determined to be converged by any of the specified
    convergence tests.

    Parameters
    ----------
    tests : args
        The :class:`ConvergenceTest`\s to be used.

    """
    def __init__(self, *tests):
        super().__init__(*tests)

    def converged(self, result):
        """Check if the function has converged by any of the specified tests.

        Parameters
        ----------
        result : :class:`~relentless.optimize.objective.ObjectiveFunctionResult`
            The location of the function at which to check for convergence.

        Returns
        -------
        bool
            ``True`` if the function is converged by any test.

        """
        return any([t.converged(result) for t in self.tests])

class AllTest(LogicTest):
    """Logic test if all specified tests return convergence.

    Check if the function is determined to be converged by all of the specified
    convergence tests.

    Parameters
    ----------
    tests : args
        The :class:`ConvergenceTest`\s to be used.

    """
    def __init__(self, *tests):
        super().__init__(*tests)

    def converged(self, result):
        """Check if the function is converged by all of the specified tests.

        Parameters
        ----------
        result : :class:`~relentless.optimize.objective.ObjectiveFunctionResult`
            The location of the function at which to check for convergence.

        Returns
        -------
        bool
            ``True`` if the function is converged by all tests.

        """
        return all([t.converged(result) for t in self.tests])
