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
            self.operations = operations
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
        if not all([isinstance(op,SimulationOperation) for op in self.operations]):
            raise TypeError('All operations must be SimulationOperations.')

        sim = self._new_instance(ensemble, potentials, directory)
        for op in self.operations:
            op(sim)
        return sim

    @property
    def operations(self):
        return self._operations

    @operations.setter
    def operations(self, ops):
        try:
            self._operations = list(ops)
        except TypeError:
            self._operations = [ops]

    def _new_instance(self, ensemble, potentials, directory):
        return SimulationInstance(ensemble,potentials,directory,**self.options)

class SimulationInstance:
    """Specific instance of a simulation and its data.

    Parameters
    ----------
    ensemble : :py:class:`Ensemble`
        Simulation ensemble. Must include values for *N* and *V* even if
        these variables fluctuate.
    potentials : :py:class:`PairMatrix`
        Matrix of tabulated potentials for each pair.
    directory : :py:class:`Directory`
        Directory for output.
    options : kwargs
        Optional arguments for the initialize, analyze, and defined "operations" functions.

    """
    def __init__(self, ensemble, potentials, directory, **options):
        self.ensemble = ensemble
        self.potentials = potentials
        self.directory = directory
        for opt,val in options.items():
            setattr(self,opt,val)
        self._opdata = {}

    def __getitem__(self, op):
        if not op in self._opdata:
            self._opdata[op] = op.Data()
        return self._opdata[op]

class SimulationOperation(abc.ABC):
    class Data:
        pass

    @abc.abstractmethod
    def __call__(self, sim):
        pass
