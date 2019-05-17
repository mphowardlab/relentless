import numpy as np

from . import core

class Mock(core.TrajectoryReader):
    def __init__(self, ensemble):
        super().__init__(None)
        self.ensemble = ensemble

    def load(self, env, step):
        L = self.ensemble.V**(1./3.)
        box = core.Box(L)

        Ntot = np.sum([self.ensemble.N[i] for i in self.ensemble.types])
        snap = core.Snapshot(Ntot, box)
        snap.positions = np.zeros((Ntot,3))
        first = 0
        for i in self.ensemble.types:
            last = first + self.ensemble.N[i]
            snap.types[first:last] = i
            first += self.ensemble.N[i]

        return core.Trajectory(snapshots=(snap,))
