__all__ = ['Simulation']

import abc

from relentless.potential import PairPotential

class Simulation(abc.ABC):
    """Ensemble simulation container.

    Abstract base class that initializes a :py:class:`Ensemble, runs a simulation,
    and analyzes the results.

    Parameters
    ----------
    operations : array_like
        Array of Python-callable functions to call during the simulation
        (defaults to `None`).
    options : kwargs
        Arguments for the initialize, analyze, and operations functions.

    """
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
        ensemble : :py:class:`Ensemble`
            Simulation ensemble. Must include values for *N* and *V* even if
            these variables fluctuate.
        potentials : :py:class:`PairMatrix`
            Matrix of tabulated potentials for each pair.

        """
        self.initialize(ensemble,potentials,self.options)
        for op in self.operations:
            op(ensemble,potentials,self.options)
        return self.analyze(ensemble,potentials,self.options)

    @abc.abstractmethod
    def initialize(self, ensemble, potentials, options):
        """Initialize the simulation."""
        pass

    @abc.abstractmethod
    def analyze(self, ensemble, potentials, options):
        """Analyze the simulation and return the ensemble."""
        pass
