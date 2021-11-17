"""
Dilute
======

The :class:`AddEnsembleAnalyzer` simulation operation for a dilute system is provided.
It can be accessed using the corresponding :class:`~relentless.simulate.generic.GenericOperation`.

.. autosummary::
    :nosignatures:

    Dilute
    AddEnsembleAnalyzer

.. autoclass:: Dilute
    :members:

.. autoclass:: AddEnsembleAnalyzer
    :members: __call__,
        extract_ensemble

"""
import numpy

from relentless.ensemble import RDF
from relentless._math import Interpolator
from . import simulate

class Dilute(simulate.Simulation):
    r"""Simulation of a dilute system.

    A dilute system, which is defined as having a low particle density, is modeled
    using the following approximation for the pairwise interparticle force and
    radial distribution function:

    .. math::

        f_{ij}(r) = -\nabla u_{ij}(r)

    .. math::

        g_{ij}(r) = e^{-\beta u_{ij}(r)}

    The key assumption is that the radial distribution function :math:`g_{ij}(r)`
    can be determined exactly from the pair potential :math:`u_{ij}(r)`.

    Running a dilute simulation can also be helpful in finding a good initial guess
    for parameter values before running a full simulation.

    """
    pass

class _NullOperation(simulate.SimulationOperation):
    """Dummy operation that eats all arguments and doesn't do anything."""
    def __init__(self, *args, **ignore):
        pass

    def __call__(self, sim):
        pass

## initializers
class Initialize(_NullOperation):
    pass
class InitializeFromFile(Initialize):
    pass
class InitializeRandomly(Initialize):
    pass

## integrators
class MinimizeEnergy(_NullOperation):
    pass
class AddMDIntegrator(_NullOperation):
    pass
class RemoveMDIntegrator(_NullOperation):
    pass
class AddBrownianIntegrator(AddMDIntegrator):
    pass
class RemoveBrownianIntegrator(RemoveMDIntegrator):
    pass
class AddLangevinIntegrator(AddMDIntegrator):
    pass
class RemoveLangevinIntegrator(RemoveMDIntegrator):
    pass
class AddVerletIntegrator(AddMDIntegrator):
    pass
class RemoveVerletIntegrator(RemoveMDIntegrator):
    pass
class Run(_NullOperation):
    pass
class RunUpTo(_NullOperation):
    pass

## analyzers
class AddEnsembleAnalyzer(simulate.SimulationOperation):
    """Analyzes the simulation ensemble and rdf."""
    def __init__(self, *args, **ignore):
        # catch options that are used by other AddEnsembleAnalyzer methods and ignore them
        pass

    def __call__(self, sim):
        r"""Creates a copy of the ensemble. The pressure parameter for the new
        ensemble is calculated as:

        .. math::

            P=k_BT\sum_i\rho_i+\frac{2}{3}\sum_i\sum_j\rho_i\rho_j\int_0^\infty drr^3f_{ij}(r)g_{ij}(r) \\

        Parameters
        ----------
        sim : :class:`~relentless.simulate.simulate.SimulationInstance`
            Instance to analyze.

        Raises
        ------
        ValueError
            If ``r`` and ``u`` are not both set in the potentials matrix.

        """
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
        """Creates an ensemble with the averaged thermodynamic properties and rdf.

        Parameters
        ----------
        sim : :class:`~relentless.simulate.simulate.Simulation`
            The simulation object.

        Returns
        -------
        :class:`~relentless.ensemble.Ensemble`
            Ensemble with averaged thermodynamic properties and rdf.

        """
        return sim[self].ensemble
