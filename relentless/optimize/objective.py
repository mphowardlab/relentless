"""
Objective Functions
===================

An :class:`ObjectiveFunction` defines an objective function to be optimized.
An objective function is a scalar value that is defined as a function of the
problem :class:`DesignVariables<DesignVariable>`, and has a specified gradient
with respect to each of the :class:`DesignVariables<DesignVariable>` as well.

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

    An :py:class:`ObjectiveFunction` defines the objective function parametrized on
    one or more adjustable :py:class:`DesignVariables<DesignVariable>`. The function
    must also have a defined value and gradient for all values of its parameters.

    """
    @abc.abstractmethod
    def compute(self):
        """Evaluates the value and gradient of the objective function.

        This method must call :py:meth:`make_result()` and return its result.

        Returns
        -------
        :py:class:`ObjectiveFunctionResult`
            The result of :py:meth:`make_result()`.

        """
        pass

    @abc.abstractmethod
    def design_variables(self):
        """Returns all :py:class:`DesignVariables<DesignVariable>` parametrized
        by the objective function.

        Returns
        -------
        array_like
            The :py:class:`DesignVariable` parameters.

        """
        pass

    def make_result(self, value, gradient):
        """Constructs a :py:class:`ObjectiveFunctionResult` to store the result
        of :py:meth:`compute()`.

        Parameters
        ----------
        value : float
            The value of the objective function.
        gradient : dict
            The gradient of the objective function. Each partial derivative is
            keyed on the :py:class:`DesignVariable` with respect to which it is taken.

        Returns
        -------
        :py:class:`ObjectiveFunctionResult`
            Object storing the value and gradient of this objective function.

        """
        return ObjectiveFunctionResult(value, gradient, self)

class ObjectiveFunctionResult:
    """Class storing the value and gradient of a :py:class:`ObjectiveFunction`.

    Parameters
    ----------
    value : float
        The value of the objective function.
    gradient : dict
        The gradient of the objective function. Each partial derivative is
        keyed on the :py:class:`DesignVariable` with respect to which it is taken.
    objective : :class:`ObjectiveFunction`
       The objective function for which this result is constructed.

    """
    def __init__(self, value, gradient, objective):
        self.value = value

        self.design_variables = objective.design_variables()
        self._gradient = _collections.FixedKeyDict(keys=self.design_variables)
        self._gradient.update(gradient)

        self.variable_values = {x : x.value for x in self.design_variables}

    def gradient(self, var):
        """The value of the gradient for a particular :py:class:`DesignVariable`
        parameter of the objective function.

        Parameters
        ----------
        var : :py:class:`DesignVariable`
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
