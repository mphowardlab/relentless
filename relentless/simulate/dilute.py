import numpy as np

from .simulate import Simulation
from relentless.core import RDF
from relentless.core import Interpolator

class Dilute(Simulation):
    """:py:class:`Simulation` container for an NVT ensemble."""
    def initialize(self, ensemble, potentials, options):
        """Initializes the Dilute simulation.

        Parameters
        ensemble : :py:class:`Ensemble`
            Simulation ensemble; must contain values for *N* and *V*.
        potentials : :py:class:`PairMatrix`
            Matrix of tabulated potentials for each pair.
        options : kwargs

        Raises
        ------
        ValueError
            If the ensemble does not have set constant values for T, V, and N for all types.

        """
        if (not ensemble.constant['T']
            or not ensemble.constant['V']
            or not all(ensemble.constant['N'][t] for t in ensemble.types)):
            raise ValueError('Dilute simulations must be run in the NVT ensemble.')

    def analyze(self, ensemble, potentials, options):
        r"""Creates a copy of the ensemble with cleared fluctuating/conjugate variables.
        The pressure *P* and :math:`g(r)` parameters for the new ensemble are calculated
        as follows:

        .. math::

            g_{ij}(r)=e^{-\beta u_{ij}(r)}

            P=k_BT\sum_i\rho_i+\frac{2}{3}\sum_i\sum_j\rho_i\rho_j\int_0^\infty drr^3f_{ij}(r)g_{ij}(r)
            f_{ij}(r)=-\nabla u_{ij}(r)

        Parameters
        ----------
        ensemble : :py:class:`Ensemble`
            Simulation ensemble; must contain values for *N* and *V*.
        potentials : :py:class:`PairMatrix`
            Matrix of tabulated potentials for each pair.
        options : kwargs

        Returns
        -------
        :py:class:`Ensemble`
            Ensemble with calculated RDF and pressure parameters.

        Raises
        ------
        ValueError
            If r and u are not both set in the potentials matrix.

        """
        new_ens = ensemble.copy()
        new_ens.clear()

        for pair in potentials:
            r = potentials[pair].get('r')
            u = potentials[pair].get('u')
            if r is None or u is None:
                raise ValueError('r and u must be set in the potentials matrix.')
            gr = np.exp(-ensemble.beta*u)
            new_ens.rdf[pair] = RDF(r,gr)

        #calculate pressure
        new_ens.P = 0.
        for a in ensemble.types:
            rho_a = ensemble.N[a]/ensemble.V.volume
            new_ens.P += ensemble.kB*ensemble.T*rho_a
            for b in ensemble.types:
                rho_b = ensemble.N[b]/ensemble.V.volume
                r = potentials[a,b].get('r')
                u = potentials[a,b].get('u')
                f = potentials[a,b].get('f')
                if f is None:
                    f = np.zeros_like(u)
                    ur = Interpolator(r,u)
                    for i,ri in enumerate(r):
                        f[i] = -ur.derivative(ri,1)
                gr = new_ens.rdf[pair].table[:,1]
                new_ens.P += (2.*np.pi/3.)*rho_a*rho_b*np.trapz(f*gr*r**3, x=r)

        return new_ens
