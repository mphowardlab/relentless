__all__ = ['Simulation','SimulationInstance']

import abc

class Simulation(abc.ABC):
    """Ensemble simulation container.

    Abstract base class that initializes a :py:class:`Ensemble, runs a simulation,
    and analyzes the results.

    Parameters
    ----------
    operations : array_like
        Array of :py:class:`Callable` functions to call during the simulation
        (defaults to `None`).
    options : kwargs
        Optional arguments for the initialize, analyze, and defined "operations" functions.

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

        A new simulation instance is created to perform the run. It is intended
        to be destroyed at the end of the run to prevent memory leaks.

        Parameters
        ----------
        ensemble : :py:class:`Ensemble`
            Simulation ensemble. Must include values for *N* and *V* even if
            these variables fluctuate.
        potentials : :py:class:`PairMatrix`
            Matrix of tabulated potentials for each pair.

        """
        sim = SimulationInstance(ensemble,potentials,**self.options)
        self.initialize(sim)
        for op in self.operations:
            op(sim)
        result = self.analyze(sim)
        del sim

        return result

    @abc.abstractmethod
    def initialize(self, sim):
        """Initialize the simulation.

        Parameters
        ----------
        sim : :py:class:`SimulationInstance`
            Instance to initialize.

        """
        pass

    @abc.abstractmethod
    def analyze(self, sim):
        """Analyze the simulation.

        Parameters
        ----------
        sim : :py:class:`SimulationInstance`
            Instance to analyze.

        """
        pass

class SimulationInstance:
    """Specific instance of a simulation and its data.

    Parameters
    ----------
    ensemble : :py:class:`Ensemble`
        Simulation ensemble. Must include values for *N* and *V* even if
        these variables fluctuate.
    potentials : :py:class:`PairMatrix`
        Matrix of tabulated potentials for each pair.
    options : kwargs
        Optional arguments for the initialize, analyze, and defined "operations" functions.

    """
    def __init__(self, ensemble, potentials, **options):
        self.ensemble = ensemble
        self.potentials = potentials
        for opt,val in options.items():
            setattr(self,opt,val)
