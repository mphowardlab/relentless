from .potential import *
from .pair import *

class ForceField:
    """Combination of multiple potentials.
    """
    def __init__(self):
        self._pair = pair.PairPotentialTabulator(None,None)

    @property
    def pair(self):
        return self._pair
