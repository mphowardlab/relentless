import numpy as np

from relentless.core import Interpolator,RDF
from . import simulate

class Dilute(simulate.Simulation):
    """Simulation of a dilute system."""
    pass

class _NullOperation(simulate.SimulationOperation):
    """Dummy operation that eats all arguments and doesn't do anything."""
    def __init__(self, **ignore):
        pass

    def __call__(self, sim):
        pass

## initializers
class Initialize(_NullOperation):
    pass
class InitializeFromFile(Initialize):
    pass
class InitiializeRandomly(Initialize):
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
# NPT integrators are not supported (only NVT)
# skipping AddNPTIntegrator / RemoveNPTIntegrator
class AddNVTIntegrator(AddMDIntegrator):
    pass
class RemoveNVTIntegrator(RemoveMDIntegrator):
    pass
class Run(_NullOperation):
    pass
class RunUpTo(_NullOperation):
    pass

## analyzers
class AddEnsembleAnalyzer(simulate.SimulationOperation):
    def __init__(self, **ignore):
        # catch options that are used by other AddEnsembleAnalyzer methods and ignore them
        pass

    def __call__(self, sim):
        r"""Creates a copy of the ensemble with cleared fluctuating/conjugate variables.
        The pressure *P* and :math:`g(r)` parameters for the new ensemble are calculated
        as follows:

        .. math::

            g_{ij}(r)=e^{-\beta u_{ij}(r)}

            P=k_BT\sum_i\rho_i+\frac{2}{3}\sum_i\sum_j\rho_i\rho_j\int_0^\infty drr^3f_{ij}(r)g_{ij}(r)

            f_{ij}(r)=-\nabla u_{ij}(r)

        Parameters
        ----------
        sim : :py:class:`SimulationInstance`
            Instance to analyze.

        Raises
        ------
        ValueError
            If r and u are not both set in the potentials matrix.

        """
        if not sim.ensemble.aka("NVT"):
            raise ValueError('Dilute simulations must be run in the NVT ensemble.')

        ens = sim.ensemble.copy()
        ens.clear()

        # pair distribution function
        for pair in sim.potentials:
            r = sim.potentials[pair].get('r')
            u = sim.potentials[pair].get('u')
            if r is None or u is None:
                raise ValueError('r and u must be set in the potentials matrix.')
            gr = np.exp(-sim.ensemble.beta*u)
            ens.rdf[pair] = RDF(r,gr)

        # compute pressure
        ens.P = 0.
        for a in ens.types:
            rho_a = ens.N[a]/ens.V.volume
            ens.P += ens.kB*ens.T*rho_a
            for b in ens.types:
                rho_b = ens.N[b]/ens.V.volume
                r = sim.potentials[a,b].get('r')
                u = sim.potentials[a,b].get('u')
                f = sim.potentials[a,b].get('f')
                if f is None:
                    ur = Interpolator(r,u)
                    f = -ur.derivative(r,1)
                gr = ens.rdf[pair].table[:,1]
                ens.P += (2.*np.pi/3.)*rho_a*rho_b*np.trapz(y=f*gr*r**3,x=r)

        sim[self].ensemble = ens

    def extract_ensemble(self, sim):
        return sim[self].ensemble
