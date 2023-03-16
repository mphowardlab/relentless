"""
Dilute
======

This module implements the :class:`Dilute` simulation backend and its specific
operations. It is best to interface with these operations using the frontend in
:mod:`relentless.simulate`.

.. autoclass:: NullOperation

"""
import numpy

from relentless.model import ensemble, extent

from . import simulate


class NullOperation(simulate.SimulationOperation):
    """Dummy operation that eats all arguments and doesn't do anything."""

    def __init__(self, *args, **ignore):
        pass

    def __call__(self, sim):
        pass


class InitializeRandomly(simulate.InitializationOperation):
    """Initializes a "randomly" generated simulation.

    Since this is a dilute system, there is not actual particle configuration.
    Instead, this operation sets up a canonical (NVT) ensemble that is
    consistent with the specified conditions.

    Parameters
    ----------
    seed : int
        The seed to randomly initialize the particle locations.
    N : dict
        Number of particles of each type.
    V : :class:`~relentless.extent.Extent`
        Simulation extent.
    T : float
        Temperature.
    masses : dict
        Masses of each particle type. These are not actually used by this
        operation, so None is the same as specifying something.
    diameters : dict
        Diameter of each particle type. These are not actually used by this
        operation, so None is the same as specifying something.

    """

    def __init__(self, seed, N, V, T, masses, diameters):
        self.N = N
        self.V = V
        self.T = T

    def __call__(self, sim):
        ens = ensemble.Ensemble(T=self.T, N=self.N, V=self.V)
        sim[self]["_ensemble"] = ens
        if isinstance(ens.V, extent.Volume):
            sim.dimension = 3
        elif isinstance(ens.V, extent.Area):
            sim.dimension = 2
        sim.types = ens.types


class _Integrator(simulate.SimulationOperation):
    def __init__(self, steps, timestep, analyzers):
        super().__init__(analyzers)
        self.steps = steps
        self.timestep = timestep


class RunBrownianDynamics(_Integrator):
    def __init__(self, steps, timestep, T, friction, seed, analyzers):
        super().__init__(steps, timestep, analyzers)
        self.T = T

    def __call__(self, sim):
        for analyzer in self.analyzers:
            analyzer.pre_run(sim, self)

        sim[sim.initializer]["_ensemble"].T = self.T

        for analyzer in self.analyzers:
            analyzer.post_run(sim, self)


class RunLangevinDynamics(_Integrator):
    def __init__(self, steps, timestep, T, friction, seed, analyzers):
        super().__init__(steps, timestep, analyzers)
        self.T = T

    def __call__(self, sim):
        for analyzer in self.analyzers:
            analyzer.pre_run(sim, self)

        sim[sim.initializer]["_ensemble"].T = self.T

        for analyzer in self.analyzers:
            analyzer.post_run(sim, self)


class RunMolecularDynamics(_Integrator):
    def __init__(self, steps, timestep, thermostat, barostat, analyzers):
        super().__init__(steps, timestep, analyzers)
        self.thermostat = thermostat
        self.barostat = barostat

    def __call__(self, sim):
        if self.barostat is not None:
            sim[sim.initializer]["_ensemble"].P = self.barostat.P

        for analyzer in self.analyzers:
            analyzer.pre_run(sim, self)

        if self.thermostat is not None:
            sim[sim.initializer]["_ensemble"].T = self.thermostat.T

        for analyzer in self.analyzers:
            analyzer.post_run(sim, self)


class EnsembleAverage(simulate.AnalysisOperation):
    r"""Analyzer for the simulation ensemble.

    The temperature, volume, and number of particles are copied from the current
    simulation ensemble. The radial distribution function is:

    .. math::

        g_{ij}(r) = e^{-\beta u_{ij}(r)}

    and the pressure is:

    .. math::

        P = k_B T \sum_i\rho_i
            - \frac{2}{3} \sum_i \sum_j \rho_i \rho_j
            \int_0^\infty drr^3 \nabla u_{ij}(r) g_{ij}(r)

    Parameters
    ----------
    check_thermo_every : int
        Number of timesteps between computing thermodynamic properties.
    check_rdf_every : int
        Number of time steps between computing the RDF.
    rdf_dr : float
        The width of a bin in the RDF histogram.

    """

    def __init__(self, check_thermo_every, check_rdf_every, rdf_dr):
        pass

    def pre_run(self, sim, sim_op):
        pass

    def post_run(self, sim, sim_op):
        pass

    def process(self, sim, sim_op):
        ens = sim[sim.initializer]["_ensemble"].copy()
        kT = sim.potentials.kB * ens.T
        # pair distribution function
        for pair in ens.pairs:
            u = sim.potentials.pair.energy(pair)
            ens.rdf[pair] = ensemble.RDF(sim.potentials.pair.x, numpy.exp(-u / kT))
        # compute pressure
        B = 0.0
        N = sum(ens.N[i] for i in sim.types)
        for i, j in sim.pairs:
            counting_factor = 2 if i != j else 1
            r = sim.potentials.pair.x
            u = sim.potentials.pair.energy((i, j))

            # only count continuous range of finite values
            flags = numpy.isinf(u)
            first_finite = 0
            while flags[first_finite] and first_finite < len(r):
                first_finite += 1
            r = r[first_finite:]
            u = u[first_finite:]

            if sim.dimension == 3:
                geo_prefactor = 4 * numpy.pi * r**2
            elif sim.dimension == 2:
                geo_prefactor = 2 * numpy.pi * r
            else:
                raise ValueError("Geometric integration factor unknown for extent type")
            y_i = ens.N[i] / N
            y_j = ens.N[j] / N
            b_ij = counting_factor * (geo_prefactor / 2.0) * (numpy.exp(-u / kT) - 1)
            B += y_i * y_j * numpy.trapz(b_ij, x=r)
        # calculate pressure if no barostat
        if ens.P is None:
            rho = N / ens.V.extent
            ens.P = kT * (rho + B * rho**2)
            sim[self]["ensemble"] = ens
            sim[self]["num_thermo_samples"] = None
            sim[self]["num_rdf_samples"] = None
        # adjust extent to maintain pressure if barostat
        else:
            coeffs = numpy.array([-ens.P / kT, N, B * N**2])
            V = numpy.max(numpy.roots(coeffs))
            L = V ** (1 / sim.dimension)

            if sim.dimension == 3:
                ens.V = extent.Cube(L)
            elif sim.dimension == 2:
                ens.V = extent.Square(L)

            sim[self]["ensemble"] = ens
            sim[self]["num_thermo_samples"] = None
            sim[self]["num_rdf_samples"] = None


class Dilute(simulate.Simulation):
    r"""Simulation of a dilute system.

    A dilute "simulation" is performed under the assumption that the radial
    distribution function :math:`g_{ij}(r)` can be determined exactly from the
    pair potential :math:`u_{ij}(r)`:

    .. math::

        g_{ij}(r) = e^{-\beta u_{ij}(r)}

    This approximation is only reasonable for low-density systems, but it can
    still quite be useful for debugging a script without needing to run a costly
    simulation. It can also be helpful for finding an initial guess for design
    variables before switching to a full simulation.

    Most of the simulation operations are :class:`NullOperation`\s that do not
    actually do anything: they will simply consume any arguments given to them.
    Only the :class:`AddEnsembleAnalyzer` operation actually implements the
    physics of the dilute simulation.

    """

    # initialize
    _InitializeRandomly = InitializeRandomly

    # md
    _MinimizeEnergy = NullOperation
    _RunBrownianDynamics = RunBrownianDynamics
    _RunLangevinDynamics = RunLangevinDynamics
    _RunMolecularDynamics = RunMolecularDynamics

    # analyze
    _EnsembleAverage = EnsembleAverage
