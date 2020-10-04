import numpy as np

from relentless.core import Interpolator,RDF
from . import simulate

class Dilute(simulate.Simulation):
    """Simulation of a dilute system.

    The :py:class:`Ensemble` must be canonical (constant *N*, *V*, and *T*).

    """
    pass

class AddEnsembleAnalyzer(simulate.AddAnalyzer):
    def __init__(self):
        super().__init__(every=None)
        self.ensemble = None

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
        new_ens = sim.ensemble.copy()
        new_ens.clear()

        for pair in sim.potentials:
            r = sim.potentials[pair].get('r')
            u = sim.potentials[pair].get('u')
            if r is None or u is None:
                raise ValueError('r and u must be set in the potentials matrix.')
            gr = np.exp(-sim.ensemble.beta*u)
            new_ens.rdf[pair] = RDF(r,gr)

        #calculate pressure
        new_ens.P = 0.
        for a in new_ens.types:
            rho_a = new_ens.N[a]/new_ens.V.volume
            new_ens.P += new_ens.kB*new_ens.T*rho_a
            for b in new_ens.types:
                rho_b = new_ens.N[b]/new_ens.V.volume
                r = sim.potentials[a,b].get('r')
                u = sim.potentials[a,b].get('u')
                f = sim.potentials[a,b].get('f')
                if f is None:
                    ur = Interpolator(r,u)
                    f = -ur.derivative(r,1)
                gr = new_ens.rdf[pair].table[:,1]
                new_ens.P += (2.*np.pi/3.)*rho_a*rho_b*np.trapz(y=f*gr*r**3,x=r)

        self.ensemble = new_ens
