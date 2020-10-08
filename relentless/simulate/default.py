import importlib
import inspect

import numpy as np

from . import simulate

class SimulationOperationAdapter(simulate.SimulationOperation):
    backends = {}

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self._op = None
        self._backend = None

    def __call__(self, sim):
        if self._op is None or self._backend != sim.backend:
            backend = self.backends.get(sim.backend)
            if not backend:
                raise TypeError('Simulation backend {} not registered.'.format(backend))

            op_name = type(self).__name__
            try:
                BackendOp = getattr(backend,op_name)
            except AttributeError:
                raise TypeError('{}.{}.{} operation not found.'.format(backend.__module__,backend.__name__,op_name))

            self._op = BackendOp(*self.args,**self.kwargs)
            self._backend = sim.backend

        return self._op(sim)

    @classmethod
    def add_backend(cls, backend, module=None):
        # try to deduce module from backend class
        if module is None:
            module = backend.__module__

        # setup module (if not already a module)
        if not inspect.ismodule(module):
            module = importlib.import_module(module)

        cls.backends[backend] = module

## initializers
class InitializeFromFile(SimulationOperationAdapter):
    pass
class InitializeRandomly(SimulationOperationAdapter):
    pass

## integrators
class MinimizeEnergy(SimulationOperationAdapter):
    pass
class AddMDIntegrator(SimulationOperationAdapter):
    pass
class RemoveMDIntegrator(SimulationOperationAdapter):
    pass
class AddBrownianIntegrator(SimulationOperationAdapter):
    pass
class RemoveBrownianIntegrator(SimulationOperationAdapter):
    pass
class AddLangevinIntegrator(SimulationOperationAdapter):
    pass
class RemoveLangevinIntegrator(SimulationOperationAdapter):
    pass
class AddNPTIntegrator(SimulationOperationAdapter):
    pass
class RemoveNPTIntegrator(SimulationOperationAdapter):
    pass
class AddNVTIntegrator(SimulationOperationAdapter):
    pass
class RemoveNVTIntegrator(SimulationOperationAdapter):
    pass
class Run(SimulationOperationAdapter):
    pass
class RunUpTo(SimulationOperationAdapter):
    pass

## analyzers
class AddEnsembleAnalyzer(SimulationOperationAdapter):
    def extract_ensemble(self, sim):
        return self._op.extract_ensemble(sim)
