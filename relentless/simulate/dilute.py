import numpy as np

from .simulate import Simulation
from relentless.core import RDF
from relentless.core import Interpolator

class Dilute(Simulation):
    """Sets up a :py:class:`Simulation` container for an NVT ensemble."""
    def initialize(self, ensemble, potentials, options):
        if (not ensemble.constant['T']
            or not ensemble.constant['V']
            or not all(ensemble.constant['N'][t] for t in ensemble.types)):
            raise ValueError('Dilute simulations must be run in the NVT ensemble.')

    def analyze(self, ensemble, potentials, options):
        r"""Creates a copy of the ensemble with cleared fluctuating/conjugate variables.
        The pressure *P* and :math:`g(r)` parameters for the new ensemble are calculated.

        """
        new_ens = ensemble.copy()
        new_ens.clear()

        for pair in potentials:
            r = potentials[pair]['r']
            u = potentials[pair]['u']
            gr = np.exp(-ensemble.beta*u)
            new_ens.rdf[pair] = RDF(r,gr)

        #calculate pressure
        rho_i_net = 0
        integral_net = 0
        for a in ensemble.types:
            rho_i_net += ensemble.N[a]/ensemble.V.volume
            for b in ensemble.types:
                rho_ij_net = ensemble.N[a]*ensemble.N[b]/(ensemble.V.volume**2)
                r = potentials[a,b]['r']
                u = potentials[a,b]['u']
                f = np.zeros_like(u)
                ur = Interpolator(r,u)
                for i,ri in enumerate(r):
                    f[i] = -ur.derivative(ri,1)
                gr = np.exp(-ensemble.beta*u)
                integral_net += (rho_ij_net*np.trapz((r**3)*f*gr, x=r))
        new_ens.P = ensemble.kB*ensemble.T*rho_i_net + (2/3)*np.pi*integral_net

        return new_ens
