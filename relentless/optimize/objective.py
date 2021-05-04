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

The following objective functions have been implemented:

.. autosummary::
    :nosignatures:

    RelativeEntropy

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

.. autoclass:: RelativeEntropy
    :member-order: bysource
    :members: compute,
        design_variables,
        target

"""
import abc

import numpy as np
import scipy.integrate

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
    r"""Relative entropy.

    The relative entropy quantifies how different two probability distributions are.
    This can be applied to the probability distributions of a target ensemble
    and the ensemble of a simulated molecular system, which is parametrized
    by some pair potential functions.

    The value of the relative entropy is not determined. The gradient of the relative
    entropy is equivalent to the gradient of the likelihood of the design variables
    :math:`\mathbf{x}` given the target ensemble :math:`\mathbf{T}`. Given
    the rdfs of the simulation and target ensembles, :math:`g(r|\mathbf{x})` and :math:`g_{tgt}(r)`,
    respectively, the total pair potential energy, the gradient is:

    .. math::

        \nabla_\mathbf{x} L\left(\mathbf{x}|\mathbf{T}\right) = 2\pi N\rho\int{dr r^2 \left(g(r|\mathbf{x})-g_{tgt}(r)\right)\beta u(r|\mathbf{x})}

    where :math:`\beta` is the thermodynamic beta, :math:`N` is the particular number,
    and :math:`\rho=N/V` is the number density, all for the target ensemble.

    Parameters
    ----------
    target : :class:`~relentless.ensemble.Ensemble`
        The target ensemble (must have specified N and V).
    simulation :class:`~relentless.simulate.Simulation`
        The simulation engine to use, with specified simulation operations.
    potentials :class:`~relentless.simulate.Potentials`
        The pair potentials to use in the simulations.
    thermo : :class:`~relentless.simulate.SimulationOperation`
        The thermodynamic analyzer for the simulation ensemble and rdf (usually
        :meth:`AddEnsembleAnalyzer()` for the specified simulation type.
    dr : float
        The radial step size.
    communicator : :class:`~relentless.mpi.Communicator`
        The communicator used for saving outputs to file ##?## (defaults to
        ``None``).

    """
    def __init__(self, target, simulation, potentials, thermo, dr, communicator=None):
        self.target = target
        self.simulation = simulation
        self.potentials = potentials
        self.thermo = thermo
        self.dr = dr
        self.communicator = communicator

    def compute(self, directory=None):
        """Evaluate the value and gradient of the relative entropy function.


        Parameters
        ----------
        directory : :class:`~relentless.data.Directory`
            The directory to which the values of the pair potential design
            variables at the time of computation are saved (defaults to ``None``).

        Returns
        -------
        :class:`ObjectiveFunctionResult`
            The result of :meth:`ObjectiveFunction.make_result()`

        """
        sim = self.simulation.run(self.target, self.potentials, directory, self.communicator)
        sim_ens = self.thermo.extract_ensemble(sim)

        # compute the relative entropy gradient by integration
        g_tgt = self.target.rdf
        g_sim = sim_ens.rdf
        dvars = self.design_variables()
        gradient = _collections.KeyedArray(keys=dvars)

        for var in dvars:
            update = 0
            for i,j in self.target.pairs:
                rs = self.potentials.pair.r
                us = self.potentials.pair.energy((i,j))
                dus = self.potentials.pair.derivative((i,j),var)

                #only count (continuous range of) finite values
                flags = np.isinf(us)
                first_finite = 0
                while not flags[first_finite] and first_finite < len(rs):
                    first_finite += 1
                rs = rs[first_finite+1:]
                dus = dus[first_finite+1:]

                #interpolate derivative wrt design variable with r
                dudvar = _math.Interpolator(rs,dus)

                # find common domain to compare rdfs
                r0 = max(g_sim[i,j].table[0][0],g_tgt[i,j].table[0][0])
                r1 = min(g_sim[i,j].table[-1][0],g_tgt[i,j].table[-1][0],self.potentials.pair.rmax)
                r = np.arange(r0,r1+0.5*self.dr,self.dr)

                # take integral by trapezoidal rule
                sim_factor = sim_ens.N[i]*sim_ens.N[j]*sim_ens.beta/(sim_ens.V.volume*self.target.V.volume)
                tgt_factor = self.target.N[i]*self.target.N[j]*self.target.beta/(self.target.V.volume**2)
                mult = 1 if i == j else 2 # 1 if same, otherwise need i,j and j,i contributions
                y = 2*mult*np.pi*r**2*(sim_factor*g_sim[i,j](r)-tgt_factor*g_tgt[i,j](r))*dudvar(r)
                update += scipy.integrate.trapz(x=r, y=y)

            gradient[var] = update

        # optionally write output to directory
        if directory is not None:
            if self.communicator is None or self.communicator.rank == self.communicator.root:
                n = 0
                for p in self.potentials.pair.potentials:
                    fname = 'potential_{}.log'.format(n)
                    with open(directory.file(fname),'w') as f:
                        p.save(f.name)
                    n += 1

        # relative entropy *value* is None
        return self.make_result(None, gradient, directory)

    def design_variables(self):
        """Return all unique, non-constant :class:`~relentless.variable.DesignVariable`\s
        parametrized by the pair potentials of the relative entropy.

        Returns
        -------
        tuple
            The :class:`~relentless.variable.DesignVariable` parameters.

        """
        dvars = set()
        for p in self.potentials.pair.potentials:
            for x in p.coeff.design_variables():
                if not x.const:
                    dvars.add(x)

        return tuple(dvars)

    @property
    def target(self):
        """:class:`~relentless.ensemble.Ensemble`: The target ensemble. Must have
        both V and N parameters set."""
        return self._target

    @target.setter
    def target(self, value):
        if value.V is None or value.N is None:
            raise ValueError('The target ensemble must have both V and N set.')
        self._target = value
