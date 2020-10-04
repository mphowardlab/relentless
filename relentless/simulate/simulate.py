__all__ = ['Simulation','SimulationInstance','SimulationOperation']

import abc

class Simulation:
    """Ensemble simulation container.

    Base class that initializes and runs a simulation described by a set of
    :py:class:`SimulationOperation`s.

    Parameters
    ----------
    operations : array_like
        Array of :py:class:`SimulationOperation` to call.
    options : kwargs
        Optional arguments to attach to each instance of a simulation.

    """
    def __init__(self, operations=None, **options):
        if operations is not None:
            try:
                self.operations = list(operations)
            except TypeError:
                self.operations = [operations]
        else:
            self.operations = []

        self.options = options

    def run(self, ensemble, potentials, directory):
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
        directory : :py:class:`Directory`
            Directory to use for writing data.

        """
        sim = SimulationInstance(ensemble,potentials,directory,**self.options)

        if not all([isinstance(op,SimulationOperation) for op in self.operations]):
            raise TypeError('Only SimulationOperations can be run by a Simulation.')
        for op in self.operations:
            op(sim)

        del sim

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
    def __init__(self, ensemble, potentials, directory, **options):
        self.ensemble = ensemble
        self.potentials = potentials
        self.directory = directory
        for opt,val in options.items():
            setattr(self,opt,val)

class SimulationOperation(abc.ABC):
    @abc.abstractmethod
    def __call__(self, sim):
        pass

class AddIntegrator(SimulationOperation):
    def __init__(self, dt):
        self.dt = dt

class Run(SimulationOperation):
    def __init__(self, steps):
        self.steps = steps

class AddAnalyzer(SimulationOperation):
    def __init__(self, every):
        self.every = every
