"""
Convergence Criteria
====================

A convergence test determines if an objective function has converged to the
desired minimum, subject to design constraints.

The following convergence tests have been implemented:

.. autosummary::
    :nosignatures:

    AllTest
    AndTest
    AnyTest
    GradientTest
    OrTest
    ValueTest

.. rubric:: Developer notes

To implement your own convergence test, create a class that derives from either of
the two abstract base classes below, and define the required properties and methods.
It may be helpful for the class to be composed having a :class:`Tolerance`.

.. autosummary::
    :nosignatures:

    ConvergenceTest
    LogicTest
    Tolerance

.. autoclass:: ConvergenceTest
    :member-order: bysource
    :members: converged

.. autoclass:: Tolerance
    :member-order: bysource
    :members: absolute,
        relative,
        isclose

.. autoclass:: GradientTest
    :member-order: bysource
    :members: tolerance,
        converged

.. autoclass:: ValueTest
    :member-order: bysource
    :members: absolute,
        relative,
        value,
        converged

.. autoclass:: LogicTest
    :member-order: bysource
    :members: converged

.. autoclass:: AllTest
    :member-order: bysource
    :members: converged

.. autoclass:: AnyTest
    :member-order: bysource
    :members: converged

.. autoclass:: AndTest
    :member-order: bysource

.. autoclass:: OrTest
    :member-order: bysource

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
            The result to check for convergence.

        """
        pass

class Tolerance:
    """Tolerance for convergence tests.

    An absolute tolerance can be any non-negative numerical value. A relative
    tolerance must be a non-negative numerical value between 0 and 1

    Parameters
    ----------
    absolute : float or dict
        The absolute tolerance or tolerances (keyed on the :class:`~relentless.optimize.objective.ObjectiveFunction`
        design variables).
    relative : float or dict
        The relative tolerance or tolerances (keyed on the :class:`~relentless.optimize.objective.ObjectiveFunction`
        design variables).

    """
    def __init__(self, absolute, relative):
        self.absolute = absolute
        self.relative = relative

    @property
    def absolute(self):
        """float or dict: The absolute tolerance(s). Must be non-negative."""
        return self._absolute

    @absolute.setter
    def absolute(self, value):
        try:
            tol = dict(value)
            err = any([t < 0 for t in value.values()])
        except TypeError:
            tol = value
            err = value < 0
        if err:
            raise ValueError('Absolute tolerances must be non-negative.')
        else:
            self._absolute = tol

    @property
    def relative(self):
        """float or dict: The relative tolerance(s). Must be between 0 and 1."""
        return self._tolerance

    @relative.setter
    def relative(self, value):
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

    def isclose(self, key, a, b):
        r"""Check if the two values are equivalent within a tolerance.

        For a specific relative tolerance :math:`t_r` and absolute tolerance :math:`t_a`,
        two values :math:`a` and :math:`b` are determined to be `close` if:

        .. math::

            \lvert a-b\rvert <= t_a+t_r\lvert b\rvert

        Parameters
        ----------
        key : :class:`~relentless.variable.DesignVariable`
            The variable on which the tolerances to be used are keyed.
        a : float
            The first value to compare.
        b : float
            The second value to compare.

        Returns
        -------
        bool
            ``True`` if values are close.

        """
        atol = self._atol(key)
        rtol = self._rtol(key)
        return np.isclose(a, b, atol=atol, rtol=rtol)

    def _atol(self, key):
        # get the scalar or keyed value of the absolute tolerance
        try:
            t = self.absolute[key]
        except TypeError:
            t = self.absolute
        except KeyError:
            raise KeyError('An absolute tolerance is not set for design variable ' + str(key))
        return t

    def _rtol(self, key):
        # get the scalar or keyed value of the relative tolerance
        try:
            t = self.relative[key]
        except TypeError:
            t = self.relative
        except KeyError:
            raise KeyError('A relative tolerance is not set for design variable ' + str(key))
        return t

class GradientTest(ConvergenceTest):
    r"""Gradient test for convergence using absolute tolerance.

    The absolute tolerance, :math:`t`, can be any non-negative numerical value.
    (The relative tolerance is automatically set to 0).

    The result is converged with respect to an unconstrained design variable
    :math:`x_i` (i.e., having :class:`~relentless.variable.DesignVariable.State` ``FREE``
    if and only if:

    .. math::

        \left\lvert\frac{\partial f}{\partial x_i}\right\rvert < t

    If an upper-bound constraint is active on :math:`x_i` (i.e. it has
    :class:`~relentless.variable.DesignVariable.State` ``HIGH``), the result
    is converged with respect to :math:`x_i` if and only if:

    .. math::

        -\frac{\partial f}{\partial x_i} > -t

    If a lower-bound constraint is active on :math:`x_i` (i.e. it has
    :class:`~relentless.variable.DesignVariable.State` ``LOW``), the result
    is converged with respect to :math:`x_i` if and only if:

    .. math::

        -\frac{\partial f}{\partial x_i} < t

    A result is converged if and only if the result is converged with respect to
    all design variables.

    Parameters
    ----------
    tolerance : float or dict
        The absolute tolerance or tolerances (keyed on the :class:`~relentless.optimize.objective.ObjectiveFunction`
        design variables).

    """
    def __init__(self, tolerance):
        self._tolerance = Tolerance(absolute=tolerance, relative=0.)

    @property
    def tolerance(self):
        """float or dict: The absolute tolerance(s). Must be non-negative."""
        return self._tolerance.absolute

    @tolerance.setter
    def tolerance(self, value):
        self._tolerance.absolute = value

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
            tol = self._tolerance._atol(x)
            if x.athigh() and  -grad < -tol:
                converged = False
                break
            elif x.atlow() and -grad > tol:
                converged = False
                break
            elif x.isfree() and np.abs(grad) > tol:
                converged = False
                break

        return converged

class ValueTest(ConvergenceTest):
    r"""Value test for convergence.

    The absolute tolerance, :math:`t`, can be any non-negative numerical value.
    A value, :math:`v`, must also be specified.

    The function is determined to be converged if :math:`\left\lvert v-x_i\right\rvert<t`
    for each of the design variables :math:`x_i`.

    Parameters
    ----------
    absolute : float or dict
        The absolute tolerance or tolerances (keyed on the :class:`~relentless.optimize.objective.ObjectiveFunction`
        design variables).
    relative : float or dict
        The relative tolerance or tolerances (keyed on the :class:`~relentless.optimize.objective.ObjectiveFunction`
        design variables).
    value : float or dict
        The value or values (keyed on the :class:`~relentless.optimize.objective.ObjectiveFunction`
        design variables) to check.

    """
    def __init__(self, absolute, relative, value):
        self._tolerance = Tolerance(absolute=absolute, relative=relative)
        self.value = value

    @property
    def absolute(self):
        """float or dict: The absolute tolerance(s). Must be non-negative."""
        return self._tolerance.absolute

    @absolute.setter
    def absolute(self, value):
        self._tolerance.absolute = value

    @property
    def relative(self):
        """float or dict: The relative tolerance(s). Must be between 0 and 1."""
        return self._tolerance.relative

    @relative.setter
    def relative(self, value):
        self._tolerance.relative = value

    @property
    def value(self):
        """float or dict: The value(s) to check."""
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    def converged(self, result):
        """Check if the function is converged using the value test.

        Determines if two the value of a result is close to the specified value
        using :meth:`Tolerance.isclose()`.

        Parameters
        ----------
        result : :class:`~relentless.optimize.objective.ObjectiveFunctionResult`
            The result to check for convergence.

        Returns
        -------
        bool
            ``True`` if the function is converged.

        """
        converged = True
        for x in result.design_variables:
            if not self._tolerance.isclose(x, x.value, self._val(x)):
                converged = False
                break

        return converged

    def _val(self, key):
        # get the scalar or keyed value of the check value
        try:
            v = self.value[key]
        except TypeError:
            v = self.value
        except KeyError:
            raise KeyError('A value to check is not set for design variable ' + str(key))
        return v

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
            The result to check for convergence.

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
            The result to check for convergence.

        Returns
        -------
        bool
            ``True`` if the function is converged by all tests.

        """
        return all([t.converged(result) for t in self.tests])

class OrTest(AnyTest):
    """Logic test if either of the specified tests return convergence.

    Check if the function is determined to be converged by either of the specified
    convergence tests.

    Parameters
    ----------
    a : :class:`ConvergenceTest`
        The first convergence test to use.
    b : :class:`ConvergenceTest`
        The second convergence test to use.

    """
    def __init__(self, a, b):
        super().__init__(a, b)

class AndTest(AllTest):
    """Logic test if both specified tests return convergence.

    Check if the function is determined to be converged by both of the specified
    convergence tests.

    Parameters
    ----------
    a : :class:`ConvergenceTest`
        The first convergence test to use.
    b : :class:`ConvergenceTest`
        The second convergence test to use.

    """
    def __init__(self, a, b):
        super().__init__(a, b)
