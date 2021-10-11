r"""
Objective functions
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

from relentless import _collections
from relentless import _math
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
            a value of ``None`` indicates no written output.

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
        a value of ``None`` indicates no written output.

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
        """:class:`~relentless._collections.KeyedArray`: The gradient of the
        objective function, keyed on its design variables."""
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
        """:class:`~relentless._collections.KeyedArray`: The design variables of
        the :class:`ObjectiveFunction` for which the result was constructed, mapped
        to the value of the variables at the time the result was constructed."""
        return self._design_variables

class RelativeEntropy(ObjectiveFunction):
    r"""Relative entropy.

    The relative entropy :math:`S_{\rm rel}` (or Kullback-Leibler divergence)
    quantifies the overlap of two probability distributions. For a known target
    statistical mechanical ensemble having distribution :math:`p_0` and a simulated
    model ensemble having distribution :math:`p` and parametrized on a set of design
    variables :math:`\mathbf{x}`, the relative entropy from the model to the target is:

    .. math::

        S_{\rm rel} = -\int d\Gamma p_0(\Gamma)\ln\left(\frac{p(\Gamma)}{p_0(\Gamma)}\right)

    where :math:`\Gamma` is an element of phase space. The relative entropy is
    zero when the two ensembles overlap completely, and it is positive otherwise.
    Hence, minimization of :math:`S_{\rm rel}` can be used to find parameters of
    a model that reproduce a known target ensemble.

    The value of the relative entropy is not readily determined in  molecular
    simulations, so this :class:`ObjectiveFunction` does not return a value.
    However, the gradient of the relative entropy with respect to the design
    variables :math:`\mathbf{x}` is much easier to compute as ensemble averages.
    Currently, the :class:`RelativeEntropy` objective function supports only
    :class:`~relentless.potential.pair.PairPotential` interactions. These interactions
    are characterized by :math:`g_{ij}(r)`, an :class:`~relentless.ensemble.RDF`
    for each pair of interacting types :math:`(i,j)` in each
    :class:`~relentless.ensemble.Ensemble`. The gradient of :math:`S_{\rm rel}` is then:

    .. math::

        \nabla_\mathbf{x} S_{\rm rel} = -\frac{1}{2}\sum_{i,j}\int{dr\left(4\pi r^2\right)\left[\frac{\beta N_i N_j}{V} g_{ij}(r)-\frac{\beta_0 N_{i,0} N_{j,0}}{V_0} g_{ij,0}(r)\right]\nabla_\mathbf{x} u_{ij}(r)}

    where :math:`\beta=1/(k_{\rm B}T)`, :math:`N_i` is the number of particles
    of type :math:`i`, :math:`V` is the volume, and :math:`u_{ij}(r)` is the pair potential
    in the *model* ensemble. The corresponding properties of the *target*
    ensemble are denoted with subscript :math:`0`.

    :math:`S_{\rm rel}` is extensive as written, meaning that it depends on the
    size of the system. This can be undesirable for optimization because it means
    certain hyperparameters are system-dependent, so the default behavior
    is to normalize :math:`s_{\rm rel}=S_{\rm rel}/V_0`. To use the extensive
    relative entropy set ``extensive=True``.

    Parameters
    ----------
    target : :class:`~relentless.ensemble.Ensemble`
        The target ensemble (must have specified ``V`` and ``N``).
    simulation : :class:`~relentless.simulate.simulate.Simulation`
        The simulation engine to use, with specified simulation operations.
    potentials : :class:`~relentless.simulate.simulate.Potentials`
        The pair potentials to use in the simulations.
    thermo : :class:`~relentless.simulate.simulate.SimulationOperation`
        The thermodynamic analyzer operation for the simulation ensemble and rdf
        (usually :meth:`~relentless.simulate.simulate.AddEnsembleAnalyzer()`).
        The model ensemble will be extracted from this operation.
    communicator : :class:`~relentless.mpi.Communicator`
        The communicator used to run the ``simulation`` (defaults to
        ``None``).
    extensive : bool
        Specification of whether the relative entropy is extensive (defaults to
        ``False``).

    """
    def __init__(self, target, simulation, potentials, thermo, communicator=None, extensive=False):
        self.target = target
        self.simulation = simulation
        self.potentials = potentials
        self.thermo = thermo
        self.communicator = communicator
        self.extensive = extensive

    def compute(self, directory=None):
        r"""Evaluate the value and gradient of the relative entropy function.

        The value of the relative entropy is not computed, but the gradient is.
        Calculating the gradient requires running a simulation, which may be
        computationally expensive.

        Optionally, a directory can be specified both to write the simulation
        output as defined in :meth:`~relentless.simulate.simulate.Simulation.run()`,
        and the values of the pair potential design variables, which are written
        to ``potential.i.json`` for the :math:`i`\th pair potential.

        Parameters
        ----------
        directory : :class:`~relentless.data.Directory`
            The ouptut directory. In addition to simulation output, the pair
            potential design variables at the time of computation are saved
            (defaults to ``None``).

        Returns
        -------
        :class:`ObjectiveFunctionResult`
            The result, which has unknown value ``None`` and known gradient.

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
                while flags[first_finite] and first_finite < len(rs):
                    first_finite += 1
                rs = rs[first_finite:]
                dus = dus[first_finite:]
                if first_finite == len(rs):
                    continue

                #interpolate derivative wrt design variable with r
                dudvar = _math.Interpolator(rs,dus)

                # find common domain to compare rdfs
                r0 = max(g_sim[i,j].domain[0],g_tgt[i,j].domain[0],dudvar.domain[0])
                r1 = min(g_sim[i,j].domain[-1],g_tgt[i,j].domain[-1],dudvar.domain[-1])
                sim_dr = np.min(np.diff(g_sim[i,j].table[:,0]))
                tgt_dr = np.min(np.diff(g_tgt[i,j].table[:,0]))
                dudvar_dr = np.min(np.diff(rs))
                dr = min(sim_dr,tgt_dr,dudvar_dr)
                r = np.arange(r0,r1+0.5*dr,dr)

                # normalization to extensive or intensive as specified
                norm_factor = self.target.V.volume if not self.extensive else 1.

                # take integral by trapezoidal rule
                sim_factor = sim_ens.N[i]*sim_ens.N[j]*sim_ens.beta/(sim_ens.V.volume*norm_factor)
                tgt_factor = self.target.N[i]*self.target.N[j]*self.target.beta/(self.target.V.volume*norm_factor)
                mult = 1 if i == j else 2 # 1 if same, otherwise need i,j and j,i contributions
                y = -2*mult*np.pi*r**2*(sim_factor*g_sim[i,j](r)-tgt_factor*g_tgt[i,j](r))*dudvar(r)
                update += scipy.integrate.trapz(y, x=r)

            gradient[var] = update

        # optionally write output to directory
        if directory is not None and (self.communicator is None or self.communicator.rank == self.communicator.root):
            for n,p in enumerate(self.potentials.pair.potentials):
                p.save(directory.file('potential.{}.json'.format(n)))

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
        r""":class:`~relentless.ensemble.Ensemble`: The target ensemble. Must have
        both ``V`` and ``N`` parameters set."""
        return self._target

    @target.setter
    def target(self, value):
        if value.V is None or value.N is None:
            raise ValueError('The target ensemble must have both V and N set.')
        self._target = value
