import numpy as np

from .simulate import Simulation
from relentless.core import RDF

class Dilute(Simulation):
    def initialize(self, ensemble, potentials, options):
        if (not ensemble.constant['T']
            or not ensemble.constant['V']
            or not all(ensemble.constant['N'][t] for t in ensemble.types)):
            raise ValueError('Dilute simulations must be run in the NVT ensemble.')

    def analyze(self, ensemble, potentials, options):
        new_ens = ensemble.copy()

        new_ens.clear()
        for pair in potentials:
            r = potentials[pair][:,0]
            u = potentials[pair][:,1]
            gr = np.exp(-ensemble.beta*u)
            new_ens.rdf[pair] = RDF(r,gr)

        return new_ens
