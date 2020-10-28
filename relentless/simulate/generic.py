import importlib
import inspect

import numpy as np

from . import simulate

class SimulationOperationAdapter(simulate.SimulationOperation):
    """
    """
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
        """
        """
        # try to deduce module from backend class
        if module is None:
            module = backend.__module__

        # setup module (if not already a module)
        if not inspect.ismodule(module):
            module = importlib.import_module(module)

        cls.backends[backend] = module

## initializers
class InitializeFromFile(SimulationOperationAdapter):
    """Initializes a simulation box and pair potentials from a file.

    Parameters
    ----------
    filename : str
        The file from which to read the system data.
    options : kwargs
        Options for file reading.
    r_buff : float
        Buffer width (defaults to 0.4).

    """
    def __init__(self, filename, r_buff=0.4, **options):
        super().__init__(filename, r_buff, **options)

class InitializeRandomly(SimulationOperationAdapter):
    """Initializes a randomly generated simulation box and pair potentials.

    Parameters
    ----------
    seed : int
        The seed to randomly initialize the particle locations (defaults to `None`).
    r_buff : float
        Buffer width (defaults to 0.4).

    """
    def __init__(self, seed=None, r_buff=0.4):
        super().__init__(seed, r_buff)

## integrators
class MinimizeEnergy(SimulationOperationAdapter):
    """:py:class:`SimulationOperation` that runs a FIRE energy minimzation until converged.

    Parameters
    ----------
    energy_tolerance : float
        Energy convergence criterion.
    force_tolerance : float
        Force convergence criterion.
    max_iterations : int
        Maximum number of iterations to run the minimization.
    dt : float
        Maximum step size.

    """
    def __init__(self, energy_tolerance, force_tolerance, max_iterations, dt):
        super().__init__(energy_tolerance, force_tolerance, max_iterations, dt)

class AddMDIntegrator(SimulationOperationAdapter):
    """:py:class:`SimulationOperation` to add an integrator (for equations of motion) to the simulation.

    Parameters
    ----------
    dt : float
        Time step size for each simulation iteration.

    """
    def __init__(self, dt):
        super().__init__(dt)

class RemoveMDIntegrator(SimulationOperationAdapter):
    """:py:class:`SimulationOperation` that removes a specified integration operation.

    Parameters
    ----------
    add_op : :py:class:`SimulationOperation`
        The addition/integration operation to be removed.

    """
    def __init__(self, add_op):
        super().__init__(add_op)

class AddBrownianIntegrator(SimulationOperationAdapter):
    """Brownian dynamics for a NVT ensemble.

    Parameters
    ----------
    dt : float
        Time step size for each simulation iteration
    friction : float
        Sets drag coefficient for each particle type.
    seed : int
        Seed used to randomly generate a uniform force.
    options : kwargs
        Options used in integrator function.

    """
    def __init__(self, dt, friction, seed, **options):
        super().__init__(dt, friction, seed, **options)

class RemoveBrownianIntegrator(SimulationOperationAdapter):
    """Removes the Brownian integrator operation.

    Parameters
    ----------
    add_op : :py:class:`AddBrownianIntegrator`
        The integrator addition operation to be removed.

    """
    def __init__(self, add_op):
        super().__init__(add_op)

class AddLangevinIntegrator(SimulationOperationAdapter):
    """Langevin dynamics for a NVT ensemble.

    Parameters
    ----------
    dt : float
        Time step size for each simulation iteration
    friction : float
        Sets drag coefficient for each particle type.
    seed : int
        Seed used to randomly generate a uniform force.
    options : kwargs
        Options used in integrator function.

    """
    def __init__(self, dt, friction, seed, **options):
        super().__init__(dt, friction, seed, **options)

class RemoveLangevinIntegrator(SimulationOperationAdapter):
    """Removes the Langevin integrator operation.

    Parameters
    ----------
    add_op : :py:class:`AddLangevinIntegrator`
        The integrator addition operation to be removed.

    """
    def __init__(self, add_op):
        super().__init__(add_op)

class AddNPTIntegrator(SimulationOperationAdapter):
    """NPT integration via MTK barostat-thermostat.

    Parameters
    ----------
    dt : float
        Time step size for each simulation iteration
    tau_T : float
        Coupling constant for the thermostat.
    tau_P : float
        Coupling constant for the barostat.
    options : kwargs
        Options used in integrator function.

    """
    def __init__(self, dt, tau_T, tau_P, **options):
        super().__init__(dt, tau_t, tau_P, **options)

class RemoveNPTIntegrator(SimulationOperationAdapter):
    """Removes the NPT integrator operation.

    Parameters
    ----------
    add_op : :py:class:`AddNPTIntegrator`
        The integrator addition operation to be removed.

    """
    def __init__(self, add_op):
        super().__init__(add_op)

class AddNVTIntegrator(SimulationOperationAdapter):
    r"""NVT integration via Nos\'e-Hoover thermostat.

    Parameters
    ----------
    add_op : :py:class:`AddNVTIntegrator`
        The integrator addition operation to be removed.

    Parameters
    ----------
    dt : float
        Time step size for each simulation iteration
    tau_T : float
        Coupling constant for the thermostat.
    options : kwargs
        Options used in integrator function.

    """
    def __init__(self, dt, tau_T, **options):
        super().__init__(dt, tau_T, **options)

class RemoveNVTIntegrator(SimulationOperationAdapter):
    """Removes the NVT integrator operation.

    Parameters
    ----------
    add_op : :py:class:`AddNVTIntegrator`
        The integrator addition operation to be removed.

    """
    def __init__(self, add_op):
        super().__init__(add_op)

class Run(SimulationOperationAdapter):
    """:py:class:`SimulationOperation` that advances the simulation by a given number of time steps.

    Parameters
    ----------
    steps : int
        Number of steps to run.

    """
    def __init__(self, steps):
        super().__init__(steps)

class RunUpTo(SimulationOperationAdapter):
    """:py:class:`SimulationOperation` that advances the simulation up to a given time step number.

    Parameters
    ----------
    step : int
        Step number up to which to run.

    """
    def __init__(self, step):
        super().__init__(step)

## analyzers
class AddEnsembleAnalyzer(SimulationOperationAdapter):
    """:py:class:`SimulationOperation` that analyzes the simulation ensemble and rdf at specified timestep intervals.

    Parameters
    ----------
    check_thermo_every : int
        Interval of time steps at which to log thermodynamic properties of the simulation.
    check_rdf_every : int
        Interval of time steps at which to log the rdf of the simulation.
    rdf_dr : float
        The width (in units *r*) of a bin in the histogram of the rdf.

    """
    def __init__(self, check_thermo_every, check_rdf_every, rdf_dr):
        super().__init__(check_thermo_every, check_rdf_every, rdf_dr)

    def extract_ensemble(self, sim):
        """Creates an ensemble with the averaged thermodynamic properties and rdf.

        Parameters
        ----------
        sim : :py:class:`Simulation`
            The simulation object.

        Returns
        -------
        :py:class:`Ensemble`
            Ensemble with averaged thermodynamic properties and rdf.
        """
        return self._op.extract_ensemble(sim)
