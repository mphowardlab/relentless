__all__ = ['Engine']

from relentless.core import PairMatrix
from relentless.potential import PairPotential

class Engine:
    trim = False

    def __init__(self, ensemble, table, potentials=None):
        self.ensemble = ensemble
        self.table = table

        # setup default potentials
        self.potentials = set()

        if potentials is not None:
            if isinstance(potentials, PairPotential):
                potentials = (potentials,)
            for p in potentials:
                self.potentials.add(p)

    def run(self, env, step):
        """Run the simulation with the current potentials."""
        raise NotImplementedError()

    def process(self, env, step):
        """Process the simulation ensemble."""
        raise NotImplementedError()

    def _tabulate_potentials(self):
        potentials = PairMatrix(self.ensemble.types)
        for pair in potentials.pairs:
            r = self.table.r
            u = self.table(pair, self.potentials)
            f = self.table.force(pair, self.potentials)
            potentials[pair] = self.table.regularize(u, f, trim=self.trim)
        return potentials
