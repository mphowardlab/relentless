import importlib
import inspect

import numpy as np

from . import simulate

class GenericOperation(simulate.SimulationOperation):
    """Generic simulation operation adapter.

    Translates a ``generic`` simulation operation into an implemented operation
    for a valid :py:class:`Simulation` backend. The backend must be an attribute
    of the :py:class:`GenericOperation`.

    Parameters
    ----------
    args : args
        Positional arguments for simulation operation.
    kwargs : kwargs
        Keyword arguments for simulation operation.

    """
    backends = {}

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self._op = None
        self._backend = None

    def __call__(self, sim):
        """Evaluates the generic simulation operation.

        Parameters
        ----------
        sim : :py:class:`Simulation`
            Simulation object.

        Returns
        -------
        :py:obj:
            The result of the generic simulation operation function.

        Raises
        ------
        TypeError
            If the specified simulation backend is not registered (using :py:meth:`add_backend`).
        TypeError
            If the specified operation is not found in the simulation backend.

        """
        if self._op is None or self._backend != sim.backend:
            backend = self.backends.get(sim.backend)
            if not backend:
                raise TypeError('Simulation backend {} not registered.'.format(backend))

            op_name = type(self).__name__
            try:
                BackendOp = getattr(backend,op_name)
            except AttributeError:
                raise TypeError('{}.{}.{} operation not found.'.format(backend.__name__,sim.backend.__name__,op_name))

            self._op = BackendOp(*self.args,**self.kwargs)
            self._backend = sim.backend

        return self._op(sim)

    @classmethod
    def add_backend(cls, backend, module=None):
        """Adds backend attribute to :py:class:`GenericOperation`.

        Parameters
        ----------
        backend : :py:class:`Simulation`
            Class to add as a backend.
        module : module or str or `None`
            Module in which the backend is defined. If `None` (default), try to
            deduce the module from ``backend.__module__``. ``module`` will be
            imported if it has not already been.

        """
        # try to deduce module from backend class
        if module is None:
            module = backend.__module__

        # setup module (if not already a module)
        if not inspect.ismodule(module):
            module = importlib.import_module(module)

        cls.backends[backend] = module

## initializers
class InitializeFromFile(GenericOperation):
    """Initializes a simulation box and pair potentials from a file.

    Parameters
    ----------
    filename : str
        The file from which to read the system data.
    neighbor_buffer : float
        Buffer width for neighbor list.
    options : kwargs
        Options for file reading.

    """
    def __init__(self, filename, neighbor_buffer, **options):
        super().__init__(filename, neighbor_buffer, **options)

class InitializeRandomly(GenericOperation):
    """Initializes a randomly generated simulation box and pair potentials.

    Parameters
    ----------
    seed : int
        The seed to randomly initialize the particle locations.
    neighbor_buffer : float
        Buffer width for neighbor list.
    options : kwargs
        Options for random initialization.

    """
    def __init__(self, seed, neighbor_buffer, **options):
        super().__init__(seed, neighbor_buffer, **options)

## integrators
class MinimizeEnergy(GenericOperation):
    """Runs an energy minimzation until converged.

    Parameters
    ----------
    energy_tolerance : float
        Energy convergence criterion.
    force_tolerance : float
        Force convergence criterion.
    max_iterations : int
        Maximum number of iterations to run the minimization.
    options : kwargs
        Options for energy minimizer.

    """
    def __init__(self, energy_tolerance, force_tolerance, max_iterations, **options):
        super().__init__(energy_tolerance, force_tolerance, max_iterations, **options)

class AddBrownianIntegrator(GenericOperation):
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

class RemoveBrownianIntegrator(GenericOperation):
    """Removes the Brownian integrator operation.

    Parameters
    ----------
    add_op : :py:class:`AddBrownianIntegrator`
        The integrator addition operation to be removed.

    """
    def __init__(self, add_op):
        super().__init__(add_op)

class AddLangevinIntegrator(GenericOperation):
    """Langevin dynamics for a NVT ensemble.

    Parameters
    ----------
    dt : float
        Time step size for each simulation iteration
    friction : float or dict
        Sets drag coefficient for each particle type (shared or per-type).
    seed : int
        Seed used to randomly generate a uniform force.
    options : kwargs
        Options used in integrator function.

    """
    def __init__(self, dt, friction, seed, **options):
        super().__init__(dt, friction, seed, **options)

class RemoveLangevinIntegrator(GenericOperation):
    """Removes the Langevin integrator operation.

    Parameters
    ----------
    add_op : :py:class:`AddLangevinIntegrator`
        The integrator addition operation to be removed.

    """
    def __init__(self, add_op):
        super().__init__(add_op)

class AddNPTIntegrator(GenericOperation):
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
        super().__init__(dt, tau_T, tau_P, **options)

class RemoveNPTIntegrator(GenericOperation):
    """Removes the NPT integrator operation.

    Parameters
    ----------
    add_op : :py:class:`AddNPTIntegrator`
        The integrator addition operation to be removed.

    """
    def __init__(self, add_op):
        super().__init__(add_op)

class AddNVTIntegrator(GenericOperation):
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

class RemoveNVTIntegrator(GenericOperation):
    """Removes the NVT integrator operation.

    Parameters
    ----------
    add_op : :py:class:`AddNVTIntegrator`
        The integrator addition operation to be removed.

    """
    def __init__(self, add_op):
        super().__init__(add_op)

class Run(GenericOperation):
    """Advances the simulation by a given number of time steps.

    Parameters
    ----------
    steps : int
        Number of steps to run.

    """
    def __init__(self, steps):
        super().__init__(steps)

class RunUpTo(GenericOperation):
    """Advances the simulation up to a given time step number.

    Parameters
    ----------
    step : int
        Step number up to which to run.

    """
    def __init__(self, step):
        super().__init__(step)

## analyzers
class AddEnsembleAnalyzer(GenericOperation):
    """Analyzes the simulation ensemble and rdf at specified timestep intervals.

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
