"""
Convergence criteria
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

import numpy

from relentless import collections
from relentless.model import variable


class ConvergenceTest(abc.ABC):
    r"""Abstract base class for optimization convergence tests.

    A :class:`ConvergenceTest` defines a test to determine if an
    :class:`~relentless.optimize.objective.ObjectiveFunction`
    :math:`f\left(\mathbf{x}\right)`, defined on a set of design variables
    :math:`\mathbf{x}=\left[x_1,\cdots,x_n\right]`, has converged to a desired point.

    """

    @abc.abstractmethod
    def converged(self, result):
        """Check if the function is converged.

        Parameters
        ----------
        result : :class:`~relentless.optimize.objective.ObjectiveFunctionResult`
            The result to check for convergence.

        Returns
        -------
        bool
            True if the result is converged.
        """
        pass


class Tolerance:
    r"""Tolerance for convergence tests.

    A tolerance can be used to check if one value is close to another in either
    an absolute or a relative sense. The test for closeness is based on the
    NumPy method :func:`numpy.isclose`, which can use both an absolute tolerance
    :math:`\varepsilon_{\rm a}` and a relative tolerance :math:`\varepsilon_{\rm r}`.

    A value :math:`a` is close to a value :math:`b` if and only if:

    .. math::

        \lvert a-b\rvert\le\varepsilon_{\rm a}+\varepsilon_{\rm r}\lvert b\rvert

    An absolute tolerance can be any non-negative numerical value. A relative
    tolerance must be a non-negative numerical value between 0 and 1. By setting
    :math:`\varepsilon_{\rm r}=0`, only an absolute tolerance test is performed.
    Similarly, setting :math:`\varepsilon_{\rm a}=0` will result in the performance
    of only a relative tolerance test.

    Parameters
    ----------
    absolute : float
        The default absolute tolerance.
    relative : float
        The default relative tolerance.

    """

    def __init__(self, absolute, relative):
        self._absolute = collections.DefaultDict(absolute)
        self._relative = collections.DefaultDict(relative)

    @property
    def absolute(self):
        """:class:`~relentless.collections.DefaultDict`: The absolute tolerance(s).
        Must be non-negative."""
        return self._absolute

    @property
    def relative(self):
        """:class:`~relentless.collections.DefaultDict`: The relative tolerance(s).
        Must be between 0 and 1."""
        return self._relative

    def isclose(self, a, b, key=None):
        """Check if the two values are equivalent within a tolerance.

        The test is performed using :func:`numpy.isclose`.

        The default :attr:`absolute` and :attr:`relative` tolerances can be overridden
        based on a ``key``, which can be any valid dictionary key other than ``None``.
        When testing for closeness, if a ``key`` is given, the keyed tolerance will be
        used if has been specified; otherwise, the default tolerance is used.

        Parameters
        ----------
        a : float
            The first value to compare.
        b : float
            The second value to compare.
        key : object
            The key to use for determining the tolerance (defaults to ``None``).

        Returns
        -------
        bool
            ``True`` if values are close.

        Raises
        ------
        ValueError
            If the absolute tolerance is not non-negative.
        ValueError
            If the relative tolerance is not between 0 and 1.

        """
        if self.absolute[key] < 0:
            raise ValueError("Absolute tolerances must be non-negative.")
        if self.relative[key] < 0 or self.relative[key] > 1:
            raise ValueError("Relative tolerances must be between 0 and 1.")
        return numpy.isclose(a, b, atol=self.absolute[key], rtol=self.relative[key])


class GradientTest(ConvergenceTest):
    r"""Gradient test for convergence using absolute tolerance.

    This test is useful for finding minima / maxima where the gradient should be zero.
    This is implemented using an absolute tolerance, :math:`\varepsilon_{\rm a}`, which
    can be any non-negative numerical value. One ``tolerance`` must be initially
    specified for all variables, but a different tolerance can be set for each variable:

    .. code::

        test = GradientTest(1.e-3, (x,y))
        test.tolerance[x] = 1.e-2

    The result is converged with respect to an unconstrained design variable
    :math:`x_i` if and only if:

    .. math::

        \left\lvert\frac{\partial f}{\partial x_i}\right\rvert < t

    If an upper-bound constraint is active on :math:`x_i`, the result
    is converged with respect to :math:`x_i` if and only if:

    .. math::

        -\frac{\partial f}{\partial x_i} > -t

    If a lower-bound constraint is active on :math:`x_i`, the result
    is converged with respect to :math:`x_i` if and only if:

    .. math::

        -\frac{\partial f}{\partial x_i} < t

    A result is converged if and only if the result is converged with respect to
    all design variables.

    Parameters
    ----------
    tolerance : float
        The default absolute tolerance.
    variables : :class:`~relentless.variable.Variable` or tuple
        Variable(s) to test convergence for in gradient.

    """

    def __init__(self, tolerance, variables):
        self._tolerance = Tolerance(absolute=tolerance, relative=0)
        self.variables = variable.graph.check_variables_and_types(
            variables, variable.Variable
        )

    @property
    def tolerance(self):
        """:class:`~relentless.collections.DefaultDict`: The absolute tolerance(s)."""
        return self._tolerance.absolute

    def converged(self, result):
        """Check if the function is converged using the absolute gradient test.

        Parameters
        ----------
        result : :class:`~relentless.optimize.objective.ObjectiveFunctionResult`
            The location of the function at which to check for convergence.

        Returns
        -------
        bool
            True if the function is converged.

        Raises
        ------
        KeyError
            If the requested variable is not in the gradient of the result.

        """
        converged = True
        for x in self.variables:
            if x not in result.gradient:
                raise KeyError("Design variable not in result")
            grad = result.gradient[x]
            tol = self.tolerance[x]
            if x.at_high() and -grad < -tol:
                converged = False
                break
            elif x.at_low() and -grad > tol:
                converged = False
                break
            elif (
                not x.at_low()
                and not x.at_high()
                and not self._tolerance.isclose(grad, 0, key=x)
            ):
                converged = False
                break

        return converged


class ValueTest(ConvergenceTest):
    r"""Value test for convergence.

    The result is converged if and only if the value of the function :math:`f`
    is close to the ``value`` according to :meth:`Tolerance.isclose`. Absolute
    and/or relative tolerances may be used.

    Parameters
    ----------
    value : float
        The value to check.
    absolute : float
        The default absolute tolerance (defaults to ``1e-8``).
    relative : float
        The default relative tolerance (defaults to ``1e-5``).

    """

    def __init__(self, value, absolute=1e-8, relative=1e-5):
        self._tolerance = Tolerance(absolute=absolute, relative=relative)
        self.value = value

    @property
    def value(self):
        """float: The value(s) to check."""
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    @property
    def absolute(self):
        """float: The absolute tolerance."""
        return self._tolerance.absolute.default

    @absolute.setter
    def absolute(self, value):
        self._tolerance.absolute.default = value

    @property
    def relative(self):
        """float: The relative tolerance."""
        return self._tolerance.relative.default

    @relative.setter
    def relative(self, value):
        self._tolerance.relative.default = value

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
            True if the function is converged.

        """
        return self._tolerance.isclose(result.value, self.value)


class LogicTest(ConvergenceTest):
    r"""Abstract base class for logical convergence tests.

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
            raise TypeError("All inputs to a LogicTest must be ConvergenceTests.")
        self.tests = tests


class AnyTest(LogicTest):
    r"""Logic test if any specified test returns convergence.

    Check if the function is determined to be converged by any of the specified
    convergence tests.

    Parameters
    ----------
    tests : args
        The :class:`ConvergenceTest`\s to be used.

    """

    def converged(self, result):
        """Check if the function has converged by any of the specified tests.

        Parameters
        ----------
        result : :class:`~relentless.optimize.objective.ObjectiveFunctionResult`
            The result to check for convergence.

        Returns
        -------
        bool
            True if the function is converged by any test.

        """
        return any(t.converged(result) for t in self.tests)


class AllTest(LogicTest):
    r"""Logic test if all specified tests return convergence.

    Check if the function is determined to be converged by all of the specified
    convergence tests.

    Parameters
    ----------
    tests : args
        The :class:`ConvergenceTest`\s to be used.

    """

    def converged(self, result):
        """Check if the function is converged by all of the specified tests.

        Parameters
        ----------
        result : :class:`~relentless.optimize.objective.ObjectiveFunctionResult`
            The result to check for convergence.

        Returns
        -------
        bool
            True if the function is converged by all tests.

        """
        return all(t.converged(result) for t in self.tests)


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
