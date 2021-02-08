__all__ = ['ObjectiveFunction','Result','RelativeEntropy']

import abc

class ObjectiveFunction(abc.ABC):
    @abc.abstractmethod
    def run(self):
        pass

    @abc.abstractmethod
    def design_variables(self):
        pass

class Result:
    def __init__(self, value, gradient):
        self.value = value
        self.gradient = gradient

class RelativeEntropy(ObjectiveFunction):
    def __init__(self, simulation, potentials, ensemble, thermo, directory):
        self.simulation = simulation
        self.potentials = potentials
        self.ensemble = ensemble
        self.thermo = thermo
        self.directory = directory

    def run(self):
        self.simulation.run(self.ensemble, self.potentials, self.directory)
        sim_ens = self.thermo.extract_ensemble()
        # relative entropy calculation
        # return Result(...)

    def design_variables(self):
        return self.potentials.design_variables()
