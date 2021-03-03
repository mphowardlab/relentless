r"""
Objective Functions
===================

An objective function is the quantity to be minimized in an optimization problem,
by adjusting the variables on which the function depends.

This function, :math:`f`, is a scalar value that is defined as a function of :math:`n`
problem :class:`DesignVariables<~relentless.variable.DesignVariable>`
:math:`\mathbf{x}=\left[x_1,\ldots,x_n\right]`.

The value of the function, :math:`f\left(\mathbf{x}\right)` is specified.
The gradient is also specified for all of the design variables:

    .. math::

        \nabla f = \left[\frac{\partial f}{\partial x_1},\ldots,\frac{\partial f}{\partial x_n}\right]

.. rubric:: Developer notes

To implement your own objective function, create a class that derives from
:class:`ObjectiveFunction` and define the required properties and methods.

.. autosummary::
    :nosignatures:

    ObjectiveFunction
    ObjectiveFunctionResult

.. autoclass:: ObjectiveFunction
    :member-order: bysource
    :members: compute,
        design_variables,
        make_result

.. autoclass:: ObjectiveFunctionResult
    :member-order: bysource
    :members: gradient

"""
import abc

from relentless import _collections

class ObjectiveFunction(abc.ABC):
    """Abstract base class for the optimization objective function.

    An :class:`ObjectiveFunction` defines the objective function parametrized on
    one or more adjustable :class:`DesignVariables<~relentless.variable.DesignVariable>`.
    The function must also have a defined value and gradient for all values of its parameters.

    """
    @abc.abstractmethod
    def compute(self):
        """Evaluate the value and gradient of the objective function.

        This method must call :meth:`make_result()` and return its result.

        Returns
        -------
        :class:`ObjectiveFunctionResult`
            The result of :meth:`make_result()`.

        """
        pass

    @abc.abstractmethod
    def design_variables(self):
        """Return all :class:`DesignVariables<~relentless.variable.DesignVariable>`
        parametrized by the objective function.

        Returns
        -------
        array_like
            The :class:`DesignVariable` parameters.

        """
        pass

    def make_result(self, value, gradient):
        """Construct a :class:`ObjectiveFunctionResult` to store the result
        of :meth:`compute()`.

        Parameters
        ----------
        value : float
            The value of the objective function.
        gradient : dict
            The gradient of the objective function. Each partial derivative is
            keyed on the :class:`~relentless.variable.DesignVariable`
            with respect to which it is taken.

        Returns
        -------
        :class:`ObjectiveFunctionResult`
            Object storing the value and gradient of this objective function.

        """
        return ObjectiveFunctionResult(value, gradient, self)

class ObjectiveFunctionResult:
    """Class storing the value and gradient of a :class:`ObjectiveFunction`.

    Parameters
    ----------
    value : float
        The value of the objective function.
    gradient : dict
        The gradient of the objective function. Each partial derivative is
        keyed on the :class:`~relentless.variable.DesignVariable`
        with respect to which it is taken.
    objective : :class:`ObjectiveFunction`
       The objective function for which this result is constructed.

    """
    def __init__(self, value, gradient, objective):
        self.value = value

        dvars = objective.design_variables()
        self._gradient = _collections.FixedKeyDict(keys=dvars)
        self._gradient.update(gradient)

        self._design_variables = _collections.FixedKeyDict(keys=dvars)
        variable_values = {x: x.value for x in dvars}
        self._design_variables.update(variable_values)

    def gradient(self, var):
        """The value of the gradient for a particular :class:`~relentless.variable.DesignVariable`
        parameter of the objective function.

        Parameters
        ----------
        var : :class:`~relentless.variable.DesignVariable`
            A parameter of the objective function.

        Returns
        -------
        float
            The value of the gradient if it is defined for the specified variable,
            or ``0.0`` if it is not.

        """
        try:
            return self._gradient[var]
        except KeyError:
            return 0.0

    @property
    def design_variables(self):
        """:class:`~relentless.FixedKeyDict`: The design variables of the
        :class:`ObjectiveFunction` for which the result was constructed, mapped
        to the value of the variables at the time the result was constructed."""
        return self._design_variables
