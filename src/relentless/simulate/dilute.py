"""
Dilute
======

This module implements the :class:`Dilute` simulation backend and its specific
operations. It is best to interface with these operations using the frontend in
:mod:`relentless.simulate`.

.. autoclass:: NullOperation

"""

import abc

import numpy

from relentless import math, mpi
from relentless.model import ensemble, extent

from . import analyze, md, simulate


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
        sim["engine"]["_ensemble"] = ens
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

    def __call__(self, sim):
        for analyzer in self.analyzers:
            analyzer.pre_run(sim, self)

        self._modify_ensemble(sim)

        for analyzer in self.analyzers:
            analyzer.post_run(sim, self)

    @abc.abstractmethod
    def _modify_ensemble(self, sim):
        pass


class RunBrownianDynamics(_Integrator):
    def __init__(self, steps, timestep, T, friction, seed, analyzers):
        super().__init__(steps, timestep, analyzers)
        self.T = T

    def _modify_ensemble(self, sim):
        thermostat = md.Thermostat(self.T)
        if thermostat.anneal:
            raise ValueError("Temperature annealing is not supported")
        sim["engine"]["_ensemble"].T = thermostat.T

        # compute pressure from EOS at current volume
        sim["engine"]["_ensemble"].P = None
        P, V = Dilute._virial_equation(sim, sim["engine"]["_ensemble"])
        sim["engine"]["_ensemble"].P = P
        sim["engine"]["_ensemble"].V = V


class RunLangevinDynamics(_Integrator):
    def __init__(self, steps, timestep, T, friction, seed, analyzers):
        super().__init__(steps, timestep, analyzers)
        self.T = T

    _modify_ensemble = RunBrownianDynamics._modify_ensemble


class RunMolecularDynamics(_Integrator):
    def __init__(self, steps, timestep, thermostat, barostat, analyzers):
        super().__init__(steps, timestep, analyzers)
        self.thermostat = thermostat
        self.barostat = barostat

    def _modify_ensemble(self, sim):
        if self.thermostat is not None:
            if self.thermostat.anneal:
                raise ValueError("Temperature annealing is not supported")
            sim["engine"]["_ensemble"].T = self.thermostat.T

        # if barostat, solve for volume. otherwise, compute pressure.
        if self.barostat:
            sim["engine"]["_ensemble"].P = self.barostat.P
            sim["engine"]["_ensemble"].V = None
        else:
            sim["engine"]["_ensemble"].P = None
        P, V = Dilute._virial_equation(sim, sim["engine"]["_ensemble"])
        sim["engine"]["_ensemble"].P = P
        sim["engine"]["_ensemble"].V = V


class EnsembleAverage(simulate.AnalysisOperation):
    def __init__(self, filename, every, rdf, assume_constraints):
        self.filename = filename
        self.every = every
        self.rdf = rdf
        self.assume_constraints = assume_constraints

    def pre_run(self, sim, sim_op):
        pass

    def post_run(self, sim, sim_op):
        pass

    def process(self, sim, sim_op):
        # pilfer ensemble from engine, we are computing it along the way
        ens = sim["engine"]["_ensemble"].copy()
        if ens.P is None or ens.V is None:
            ens.P, ens.V = Dilute._virial_equation(sim, ens)

        # optionally finalize RDF using given parameters
        rdf_params = self._get_rdf_params(sim)
        if rdf_params is not None:
            kT = sim.potentials.kB * ens.T
            for pair in sim.pairs:
                bin_edges = numpy.linspace(
                    0, rdf_params["stop"], rdf_params["bins"] + 1
                )
                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                u = sim.potentials.pair.energy(pair, x=bin_centers)
                ens.rdf[pair] = ensemble.RDF(bin_centers, numpy.exp(-u / kT))

        sim[self]["ensemble"] = ens
        sim[self]["num_thermo_samples"] = None
        sim[self]["num_rdf_samples"] = None

        # optionally save file
        if self.filename is not None:
            if mpi.world.rank_is_root:
                ens.save(sim.directory.file(self.filename))
            mpi.world.barrier()

    _get_rdf_params = analyze.EnsembleAverage._get_rdf_params


class Dilute(simulate.Simulation):
    r"""Simulation of a dilute system.

    A dilute "simulation" is performed under the assumption that the radial
    distribution function :math:`g_{ij}(r)` can be determined exactly from the
    pair potential :math:`u_{ij}(r)`:

    .. math::

        g_{ij}(r) = e^{-\beta u_{ij}(r)}

    The virial equation of state is used to calculate the pressure. The second
    virial coefficient (B) is computed as:

    .. math::

        B = \sum_i \sum_j x_i x_j B_{ij}

    where :math:`x_i = N_i/N` is the mole fraction of component *i*
    (:math:`N = \sum_i N_i`) and

    .. math::

        B_{ij} = -\frac{1}{2}
        \int {\rm d}\mathbf{r} [e^{-\beta u_{ij}(|\mathbf{r}|)} - 1]

    Using the number density and second virial coefficient, the pressure is:

    .. math::

        P = k_B T (\rho + B\rho^2)

    This approximation is only reasonable for low-density systems, but it can
    still quite be useful for debugging a script without needing to run a costly
    simulation. It can also be helpful for finding an initial guess for design
    variables before switching to a full simulation.

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

    @staticmethod
    def _virial_equation(sim, ens):
        if ens.T is None:
            raise ValueError("Temperature must be specified")
        kT = sim.potentials.kB * ens.T

        N = sum(ens.N[i] for i in sim.types)
        B = 0.0
        for i, j in sim.pairs:
            r = sim.potentials.pair.linear_space
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
            B_ij = -0.5 * math._trapezoid(geo_prefactor * (numpy.exp(-u / kT) - 1), r)

            y_i = ens.N[i] / N
            y_j = ens.N[j] / N
            counting_factor = 2 if i != j else 1
            B += counting_factor * y_i * y_j * B_ij

        if ens.P is None:
            # calculate pressure if no barostat
            rho = N / ens.V.extent
            P = kT * (rho + B * rho**2)
            V = ens.V
        else:
            # adjust extent to maintain pressure if barostat
            coeffs = numpy.array([-ens.P / kT, 1, B])
            roots = numpy.roots(coeffs)
            if not numpy.allclose(numpy.imag(roots), 0):
                raise ValueError(
                    "Unable to solve for volume, imaginary roots obtained."
                )
            v = numpy.max(numpy.real(roots))
            L = (v * N) ** (1 / sim.dimension)

            P = ens.P
            if sim.dimension == 3:
                V = extent.Cube(L)
            elif sim.dimension == 2:
                V = extent.Square(L)

        return P, V
