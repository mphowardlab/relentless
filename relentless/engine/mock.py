import numpy as np

from relentless.engine import Engine
from relentless.core import RDF

class Mock(Engine):
    def __init__(self, ensemble, table, potentials=None):
        super().__init__(ensemble, table, potentials)

    def run(self, env, step):
        with env.data(step):
            potentials = self._tabulate_potentials()
            for i,j in potentials:
                file_ = 'pair.{i}.{j}.dat'.format(i=i,j=j)
                np.savetxt(file_, potentials[(i,j)], header='r u f')

    def process(self, env, step):
        ens = self.ensemble.copy()

        # use dilute g(r) approximation
        potentials = self._tabulate_potentials()
        for pair in potentials:
            r = potentials[pair][:,0]
            u = potentials[pair][:,1]
            gr = np.exp(-ens.beta*u)
            ens.rdf[pair] = RDF(r,gr)

        return ens
