r"""
Objective Functions
===================

An objective function is the quantity to be minimized in an optimization problem,
by adjusting the variables on which the function depends.

This function, :math:`f`, is a scalar value that is defined as a function of :math:`n`
problem :class:`~relentless.variable.DesignVariable`\s :math:`\mathbf{x}=\left[x_1,\ldots,x_n\right]`.

The value of the function, :math:`f\left(\mathbf{x}\right)` is specified.
The gradient is also specified for all of the design variables:

    .. math::

        \nabla f\left(\mathbf{x}\right) = \left[\frac{\partial f}{\partial x_1},
                                                \ldots,
                                                \frac{\partial f}{\partial x_n}\right]

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
    :members: value,
        gradient,
        design_variables,
        directory

"""
import abc

import numpy as np
import scipy

from relentless import _collections, _math
from relentless import data

class ObjectiveFunction(abc.ABC):
    """Abstract base class for the optimization objective function.

    An :class:`ObjectiveFunction` defines the objective function parametrized on
    one or more adjustable :class:`~relentless.variable.DesignVariable`\s.
    The function must also have a defined value and gradient for all values of its parameters.

    """
    @abc.abstractmethod
    def compute(self, directory=None):
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
        """Return all :class:`~relentless.variable.DesignVariable`\s
        parametrized by the objective function.

        Returns
        -------
        array_like
            The :class:`~relentless.variable.DesignVariable` parameters.

        """
        pass

    def make_result(self, value, gradient, directory):
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
        directory : :class:`~relentless.data.Directory`
            Directory holding written output associated with result. Setting
            a value of `None` indicates no written output.

        Returns
        -------
        :class:`ObjectiveFunctionResult`
            Object storing the value and gradient of this objective function.

        """
        return ObjectiveFunctionResult(self, value, gradient, directory)

class ObjectiveFunctionResult:
    """Class storing the value and gradient of a :class:`ObjectiveFunction`.

    Parameters
    ----------
    objective : :class:`ObjectiveFunction`
       The objective function for which this result is constructed.
    value : float
        The value of the objective function.
    gradient : dict
        The gradient of the objective function. Each partial derivative is
        keyed on the :class:`~relentless.variable.DesignVariable`
        with respect to which it is taken.
    directory : :class:`~relentless.data.Directory`
        Directory holding written output associated with result. Setting
        a value of `None` indicates no written output.

    """
    def __init__(self, objective, value, gradient, directory):
        dvars = objective.design_variables()
        self._design_variables = _collections.KeyedArray(keys=dvars)
        variable_values = {x: x.value for x in dvars}
        self._design_variables.update(variable_values)

        self._value = value

        self._gradient = _collections.KeyedArray(keys=dvars)
        self._gradient.update(gradient)

        self.directory = directory

    @property
    def value(self):
        """float: The value of the evaluated objective function."""
        return self._value

    @property
    def gradient(self):
        """:class:`~relentless.KeyedArray`: The gradient of the objective function,
        keyed on its design variables."""
        return self._gradient

    @property
    def directory(self):
        """:class:`~relentless.data.Directory` Directory holding written output."""
        return self._directory

    @directory.setter
    def directory(self, value):
        if value is not None and not isinstance(value, data.Directory):
            value = data.Directory(value)
        self._directory = value

    @property
    def design_variables(self):
        """:class:`~relentless.KeyedArray`: The design variables of the
        :class:`ObjectiveFunction` for which the result was constructed, mapped
        to the value of the variables at the time the result was constructed."""
        return self._design_variables

class RelativeEntropy(ObjectiveFunction):
    def __init__(self, target, simulation, potentials, thermo, dr, communicator=None):
        self.target = target
        self.simulation = simulation
        self.potentials = potentials
        self.thermo = thermo
        self.dr = dr
        self.communicator = communicator

    def compute(self, directory=None):
        sim = self.simulation.run(self.target, self.potentials, directory, self.communicator)
        sim_ens = self.thermo.extract_ensemble(sim)

        # compute the relative entropy gradient by integration
        g_tgt = self.target.rdf
        g_sim = sim_ens.rdf
        gradient = {}
        dvars = self.design_variables()

        for x in dvars:
            update = 0
            for i,j in self.potentials.pair.potentials.coeff.pairs:
                # evaluate derivative wrt x and interpolate
                dudx = _math.Interpolator(self.potentials.pair.x, self.potentials.pair.derivative((i,j), x))

                # find common domain to compare rdfs
                r0 = max(g_sim[i,j].table[0][0], g_tgt[i,j].table[0][0])
                r1 = min(g_sim[i,j].table[-1][0], g_tgt[i,j].table[-1][0])
                r = np.arange(r0, r1+0.5*self.dr, self.dr)

                # take integral by trapezoidal rule
                sim_factor = sim_ens.N[i]*sim_ens.N[j]/sim_ens.V.volume
                tgt_factor = self.target.N[i]*self.target.N[j]/self.target.V.volume
                mult = 2 if i == j else 4 # 2 if same, otherwise need i,j and j,i contribution
                y = mult*np.pi*r**2*(sim_factor*g_sim[i,j](r)-tgt_factor*g_tgt[i,j](r))*self.target.beta*dudx(r)
                update += scipy.integrate.trapz(x=r, y=y)

            gradient[x] = update

        #TODO: directory

        return self.make_result(None, gradient, directory)

    def design_variables(self):
        dvars = set()
        for p in self.potentials.pair.potentials:
            for x in p.coeff.design_variables():
                if not x.const:
                    dvars.add(x)

        return tuple(dvars)
