"""
Dilute
======

This module implements the :class:`Dilute` simulation backend and its specific
operations. It is best to interface with these operations using the frontend in
:mod:`relentless.simulate`.

.. autoclass:: NullOperation

"""
import numpy

from relentless.ensemble import RDF
from relentless import extent
from . import simulate

class NullOperation(simulate.SimulationOperation):
    """Dummy operation that eats all arguments and doesn't do anything."""
    def __init__(self, *args, **ignore):
        pass

    def __call__(self, sim):
        pass

class AddEnsembleAnalyzer(simulate.SimulationOperation):
    r"""Analyzer for the simulation ensemble.

    The temperature, volume, and number of particles are copied from the current
    simulation ensemble. The radial distribution function is:

    .. math::

        g_{ij}(r) = e^{-\beta u_{ij}(r)}

    and the pressure is:

    .. math::

        P=k_BT\sum_i\rho_i-\frac{2}{3}\sum_i\sum_j\rho_i\rho_j\int_0^\infty drr^3 \nabla u_{ij}(r) g_{ij}(r)

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

    def __call__(self, sim):
        ens = sim.ensemble.copy()

        # pair distribution function
        for pair in ens.pairs:
            u = sim.potentials.pair.energy(pair)
            ens.rdf[pair] = RDF(sim.potentials.pair.r,
                                numpy.exp(-sim.ensemble.beta*u))

        # compute pressure
        ens.P = 0.
        for a in ens.types:
            rho_a = ens.N[a]/ens.V.extent
            ens.P += ens.kT*rho_a
            for b in ens.types:
                rho_b = ens.N[b]/ens.V.extent
                r = sim.potentials.pair.r
                u = sim.potentials.pair.energy((a,b))
                f = sim.potentials.pair.force((a,b))
                gr = ens.rdf[a,b].table[:,1]

                # only count continuous range of finite values
                flags = numpy.isinf(u)
                first_finite = 0
                while flags[first_finite] and first_finite < len(r):
                    first_finite += 1
                r = r[first_finite:]
                u = u[first_finite:]
                f = f[first_finite:]
                gr = gr[first_finite:]

                if isinstance(ens.V, extent.Volume):
                    geo_prefactor = 4*numpy.pi*r**2
                elif isinstance(ens.V, extent.Area): 
                    geo_prefactor = 2*numpy.pi*r
                else:
                    raise ValueError('Geometric integration factor unknown for extent type')
                y = (geo_prefactor/6.)*rho_a*rho_b*f*gr*r
                ens.P += numpy.trapz(y,x=r)
    
        sim[self].ensemble = ens

    def extract_ensemble(self, sim):
        """Extract the average ensemble from the simulation.

        The "average" ensemble is the result of the most recent
        :meth:`__call__`.

        Parameters
        ----------
        sim : :class:`~relentless.simulate.simulate.SimulationInstance`
            The simulation.

        Returns
        -------
        :class:`~relentless.ensemble.Ensemble`
            Average ensemble from the simulation data.

        """
        return sim[self].ensemble

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
    # initialization
    InitializeFromFile = NullOperation
    InitializeRandomly = NullOperation

    # energy minimization
    MinimizeEnergy = NullOperation

    # md integrators
    AddBrownianIntegrator = NullOperation
    RemoveBrownianIntegrator = NullOperation
    AddLangevinIntegrator = NullOperation
    RemoveLangevinIntegrator = NullOperation
    AddVerletIntegrator = NullOperation
    RemoveVerletIntegrator = NullOperation

    # run commands
    Run = NullOperation
    RunUpTo = NullOperation

    # analysis
    AddEnsembleAnalyzer = AddEnsembleAnalyzer
