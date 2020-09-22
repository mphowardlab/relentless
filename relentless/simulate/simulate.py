__all__ = ['Simulation']

import abc

from relentless.potential import PairPotential

class Simulation(abc.ABC):
    default_options = {}

    def __init__(self, operations=None, **options):
        if operations is not None:
            self.operations = list(operations)
        else:
            self.operations = []

        # load up options, including user overrides
        self.options = dict(self.default_options)
        self.options.update(options)

    def run(self, ensemble, potentials):
        """Run the simulation and return the result of analyze.

        Parameters
        ----------
        ensemble : :py:class:`relentless.Ensemble`
            Simulation ensemble. Must include values for *N* and *V* even if
            these variables fluctuate.
        potentials : :py:class:`relentless.core.PairMatrix`
            Matrix of tabulated potentials for each pair.

        """
        self.initialize(ensemble,potentials,options)
        for op in self.operations:
            op(ensemble,potentials,options)
        return self.analyze(ensemble,potentials,options)

    @abc.abstractmethod
    def initialize(self, ensemble, potentials, options):
        """Initialize the simulation."""
        pass

    @abc.abstractmethod
    def analyze(self, ensemble, potentials, options):
        """Analyze the simulation and return the ensemble."""
        pass
