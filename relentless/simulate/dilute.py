"""
Dilute
======

The :class:`AddEnsembleAnalyzer` simulation operation for a dilute system is provided.
It can be accessed using the corresponding :class:`~relentless.simulate.generic.GenericOperation`.

.. autosummary::
    :nosignatures:

    Dilute
    AddEnsembleAnalyzer
    NullOperation

.. autoclass:: Dilute

.. autoclass:: AddEnsembleAnalyzer
    :members: extract_ensemble

.. autoclass:: NullOperation

"""
import numpy

from relentless.ensemble import RDF
from . import simulate

class NullOperation(simulate.SimulationOperation):
    """Dummy operation that eats all arguments and doesn't do anything."""
    def __init__(self, *args, **ignore):
        pass

    def __call__(self, sim):
        pass

class AddEnsembleAnalyzer(simulate.SimulationOperation):
    r"""Analyzes the simulation ensemble and rdf.

    The temperature, volume, and number of particles are copied from the
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
            rho_a = ens.N[a]/ens.V.volume
            ens.P += ens.kT*rho_a
            for b in ens.types:
                rho_b = ens.N[b]/ens.V.volume
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

                ens.P += (2.*numpy.pi/3.)*rho_a*rho_b*numpy.trapz(y=f*gr*r**3,x=r)

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
