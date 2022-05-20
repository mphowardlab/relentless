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
    :members: compute

.. autoclass:: ObjectiveFunctionResult
    :member-order: bysource
    :members: variables,
        value,
        gradient,
        directory

.. autoclass:: RelativeEntropy
    :member-order: bysource
    :members: compute,
        compute_gradient,
        target

"""
import abc

import numpy
import scipy.integrate

from relentless import data
from relentless import math
from relentless import mpi
from relentless import variable

class ObjectiveFunction(abc.ABC):
    """Abstract base class for the optimization objective function.

    An :class:`ObjectiveFunction` defines the objective function parametrized on
    one or more adjustable :class:`~relentless.variable.DesignVariable`\s.
    The function must also have a defined value and gradient for all values of its parameters.

    """
    @abc.abstractmethod
    def compute(self, variables, directory=None):
        """Evaluate the value and gradient of the objective function.

        Parameters
        ----------
        variables : :class:`~relentless.variable.Variable` or tuple
            Variables to record in result.
        directory : str or :class:`~relentless.data.Directory`
            The ouptut directory. In addition to simulation output, the pair
            potential design variables at the time of computation are saved
            (defaults to ``None``).

        Returns
        -------
        :class:`ObjectiveFunctionResult`
            The result of the function.

        """
        pass

class ObjectiveFunctionResult:
    """Class storing the value and gradient of a :class:`ObjectiveFunction`.

    Parameters
    ----------
    variables : :class:`~relentless.variable.Variable` or tuple
        Variables to stash values
    value : float
        The value of the objective function.
    gradient : dict
        The gradient of the objective function. Each partial derivative is
        keyed on the :class:`~relentless.variable.DesignVariable`
        with respect to which it is taken.
    directory : :class:`~relentless.data.Directory`
        Directory holding written output associated with result. Setting
        a value of ``None`` indicates no written output.

    Raises
    ------
    KeyError
        If both ``variables`` and ``gradient`` are defined but their keys
        don't match.

    """
    def __init__(self, variables=None, value=None, gradient=None, directory=None):
        self.variables = variables
        self.value = value
        self.gradient = gradient
        self.directory = directory

    @property
    def variables(self):
        """:class:`~relentless.math.KeyedArray`: Recorded variables of the
        :class:`ObjectiveFunction`."""
        return getattr(self, '_variables', None)

    @variables.setter
    def variables(self, value):
        value = variable.graph.check_variables_and_types(value, variable.Variable)
        if len(value) > 0:
            variables_ = math.KeyedArray(keys=value)
            variables_.update({x: x.value for x in value})
        else:
            variables_ = None
        self._assert_keys_match(variables_, self.gradient)
        self._variables = variables_

    @property
    def value(self):
        """float: The value of the evaluated objective function."""
        return getattr(self, '_value', None)

    @value.setter
    def value(self, x):
        self._value = x

    @property
    def gradient(self):
        """:class:`~relentless.math.KeyedArray`: The gradient of the objective
        function, keyed on its design variables."""
        return getattr(self, '_gradient', None)

    @gradient.setter
    def gradient(self, value):
        if value is not None:
            gradient_ = math.KeyedArray(keys=value.keys())
            gradient_.update(value)
        else:
            gradient_ = None
        self._assert_keys_match(self.variables, gradient_)
        self._gradient = gradient_

    @property
    def directory(self):
        """:class:`~relentless.data.Directory` Directory holding written output."""
        return getattr(self, '_directory', None)

    @directory.setter
    def directory(self, value):
        if value is not None:
            value = data.Directory.cast(value)
        self._directory = value

    def _assert_keys_match(self, vars, grad):
        """Assert that the keys of the variables and gradient match.
        
        Parameters
        ----------
        vars : dict
            Variable dictionary-like object
        grad : dict
            Gradient dictionary-like object.
        
        Raises
        ------
        AssertionError
            If the keys of ``vars`` and ``grad`` do not match.

        """
        if vars is not None and grad is not None:
            if vars.keys() != grad.keys():
                raise AssertionError('Variable and gradient keys do not match!')

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
        (usually :class:`~relentless.simulate.simulate.AddEnsembleAnalyzer`).
        The model ensemble will be extracted from this operation.
    extensive : bool
        Specification of whether the relative entropy is extensive (defaults to
        ``False``).

    """
    def __init__(self, target, simulation, potentials, thermo, extensive=False):
        self.target = target
        self.simulation = simulation
        self.potentials = potentials
        self.thermo = thermo
        self.extensive = extensive

    def compute(self, variables, directory=None):
        r"""Evaluate the value and gradient of the relative entropy function.

        The value of the relative entropy is not computed, but the gradient is.
        Calculating the gradient requires running a simulation, which may be
        computationally expensive.

        Optionally, a directory can be specified to write the simulation output
        as defined in :meth:`~relentless.simulate.simulate.Simulation.run()`,
        namely the simulation-generated ensemble, which is written to
        ``ensemble.json``, and the values of the pair potential design variables,
        which are written to ``pair_potential.i.json`` for the :math:`i`\th pair
        potential.

        Parameters
        ----------
        variables : :class:`~relentless.variable.Variable` or tuple
            Variables with respect to which to compute gradient.
        directory : str or :class:`~relentless.data.Directory`
            The ouptut directory. In addition to simulation output, the pair
            potential design variables at the time of computation are saved
            (defaults to ``None``).

        Returns
        -------
        :class:`ObjectiveFunctionResult`
            The result, which has unknown value ``None`` and known gradient.

        """
        # run simulation and use result to compute gradient
        sim = self.simulation.run(self.target, self.potentials, directory)
        sim_ens = self.thermo.extract_ensemble(sim)
        gradient = self.compute_gradient(sim_ens, variables)

        # optionally write output to directory
        if directory is not None:
            directory = data.Directory.cast(directory)
            if mpi.world.rank_is_root:
                for n,p in enumerate(self.potentials.pair.potentials):
                    p.save(directory.file('pair_potential.{}.json'.format(n)))
                sim_ens.save(directory.file('ensemble.json'))

        # relative entropy *value* is None
        return ObjectiveFunctionResult(variables, None, gradient, directory)

    def compute_gradient(self, ensemble, variables):
        """Computes the relative entropy gradient for an ensemble.

        Parameters
        ----------
        ensemble : :class:`~relentless.ensemble.Ensemble`
            The ensemble for which to evaluate the gradient.
        variables : :class:`~relentless.variable.Variable` or tuple
            Variables with respect to which to compute gradient.

        Returns
        -------
        :class:`~relentless.math.KeyedArray`
            The gradient, keyed on the :class:`~relentless.variable.DesignVariable`\s.

        """
        # compute the relative entropy gradient by integration
        g_tgt = self.target.rdf
        g_sim = ensemble.rdf
        dvars = variable.graph.check_variables_and_types(variables, variable.Variable)
        gradient = math.KeyedArray(keys=dvars)

        for var in dvars:
            update = 0
            for i,j in self.target.pairs:
                rs = self.potentials.pair.r
                us = self.potentials.pair.energy((i,j))
                dus = self.potentials.pair.derivative((i,j),var)

                #only count (continuous range of) finite values
                flags = numpy.isinf(us)
                first_finite = 0
                while flags[first_finite] and first_finite < len(rs):
                    first_finite += 1
                rs = rs[first_finite:]
                dus = dus[first_finite:]
                if first_finite == len(rs):
                    continue

                #interpolate derivative wrt design variable with r
                dudvar = math.Interpolator(rs,dus)

                # find common domain to compare rdfs
                r0 = max(g_sim[i,j].domain[0],g_tgt[i,j].domain[0],dudvar.domain[0])
                r1 = min(g_sim[i,j].domain[-1],g_tgt[i,j].domain[-1],dudvar.domain[-1])
                sim_dr = numpy.min(numpy.diff(g_sim[i,j].table[:,0]))
                tgt_dr = numpy.min(numpy.diff(g_tgt[i,j].table[:,0]))
                dudvar_dr = numpy.min(numpy.diff(rs))
                dr = min(sim_dr,tgt_dr,dudvar_dr)
                r = numpy.arange(r0,r1+0.5*dr,dr)

                # normalization to extensive or intensive as specified
                norm_factor = self.target.V.volume if not self.extensive else 1.

                # take integral by trapezoidal rule
                sim_factor = ensemble.N[i]*ensemble.N[j]*ensemble.beta/(ensemble.V.volume*norm_factor)
                tgt_factor = self.target.N[i]*self.target.N[j]*self.target.beta/(self.target.V.volume*norm_factor)
                mult = 1 if i == j else 2 # 1 if same, otherwise need i,j and j,i contributions
                y = -2*mult*numpy.pi*r**2*(sim_factor*g_sim[i,j](r)-tgt_factor*g_tgt[i,j](r))*dudvar(r)
                update += scipy.integrate.trapz(y, x=r)

            gradient[var] = update

        return gradient

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
