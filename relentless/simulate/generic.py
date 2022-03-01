"""
Generic simulation operations
=============================

A common set of generic simulation operations for all the compatible simulation
interfaces (HOOMD-blue, LAMMPS, dilute system) is provided. This grants the user
immense ease of use as a single command can be used to achieve the same function
using multiple simulation packages. The supported generic operations are initialization
of a system from file or randomly, energy minimization, simulation run, and a number
of integrators (Verlet, Langevin, Brownian), as well as an operation to extract
the model ensemble and RDF at given timestep intervals.

The generic simulation operations are compatible with the following simulation packages:

.. toctree::
    :maxdepth: 1

    dilute
    hoomd
    lammps

The following generic simulation operations have been implemented:

.. autosummary::
    :nosignatures:

    InitializeFromFile
    InitializeRandomly
    MinimizeEnergy
    AddBrownianIntegrator
    RemoveBrownianIntegrator
    AddLangevinIntegrator
    RemoveLangevinIntegrator
    AddVerletIntegrator
    RemoveVerletIntegrator
    Run
    RunUpTo
    AddEnsembleAnalyzer

.. rubric:: Developer notes

To implement your own generic operation, create a class that derives from :class:`GenericOperation`
and define the required methods. A backend is used to translate the generic operation
called by the user into an operation associated with a valid :class:`~relentless.simulate.simulate.SimulationOperation` type (i.e. HOOMD, LAMMPS, or dilute).

.. autosummary::
    :nosignatures:

    GenericOperation

.. autoclass:: GenericOperation
    :members: __call__,
        add_backend
.. autoclass:: InitializeFromFile
    :members:
.. autoclass:: InitializeRandomly
    :members:
.. autoclass:: MinimizeEnergy
    :members:
.. autoclass:: AddBrownianIntegrator
    :members:
.. autoclass:: RemoveBrownianIntegrator
    :members:
.. autoclass:: AddLangevinIntegrator
    :members:
.. autoclass:: RemoveLangevinIntegrator
    :members:
.. autoclass:: AddVerletIntegrator
    :members:
.. autoclass:: RemoveVerletIntegrator
    :members:
.. autoclass:: Run
    :members:
.. autoclass:: RunUpTo
    :members:
.. autoclass:: AddEnsembleAnalyzer
    :members:

"""
import importlib
import inspect

import numpy

from . import simulate

class GenericOperation(simulate.SimulationOperation):
    """Generic simulation operation adapter.

    Translates a generic simulation operation into an implemented operation
    for a valid :class:`~relentless.simulate.simulate.Simulation` backend. The
    backend must be an attribute of the :class:`GenericOperation`.

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
        sim : :class:`~relentless.simulate.simulate.Simulation`
            Simulation object.

        Returns
        -------
        :class:`object`
            The result of the generic simulation operation function.

        Raises
        ------
        TypeError
            If the specified simulation backend is not registered (using :meth:`add_backend()`).
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
        """Adds backend attribute to :class:`GenericOperation`.

        Parameters
        ----------
        backend : :class:`~relentless.simulate.simulate.Simulation`
            Class to add as a backend.
        module : module or :class:`str` or ``None``
            Module in which the backend is defined. If ``None`` (default), try to
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
    options : kwargs
        Options for file reading.

    """
    def __init__(self, filename, **options):
        super().__init__(filename, **options)

class InitializeRandomly(GenericOperation):
    """Initializes a randomly generated simulation box and pair potentials.

    Parameters
    ----------
    seed : int
        The seed to randomly initialize the particle locations.
    options : kwargs
        Options for random initialization.

    """
    def __init__(self, seed, **options):
        super().__init__(seed, **options)

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
    options : dict
        Additional options for energy minimizer (defaults to ``None``).

    """
    def __init__(self, energy_tolerance, force_tolerance, max_iterations, options=None):
        super().__init__(energy_tolerance, force_tolerance, max_iterations, options)

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
    add_op : :class:`AddBrownianIntegrator`
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
    add_op : :class:`AddLangevinIntegrator`
        The integrator addition operation to be removed.

    """
    def __init__(self, add_op):
        super().__init__(add_op)

class AddVerletIntegrator(GenericOperation):
    """Family of Verlet integration modes.

    Parameters
    ----------
    dt : float
        Time step size for each simulation iteration.
    thermostat : :class:`~relentless.simulate.simulate.Thermostat`
        Thermostat used for integration (defaults to ``None``).
    barostat : :class:`~relentless.simulate.simulate.Barostat`
        Barostat used for integration (defaults to ``None``).
    options : kwargs
        Options used in appropriate integrator function.

    """
    def __init__(self, dt, thermostat=None, barostat=None, **options):
        super().__init__(dt, thermostat, barostat, **options)

class RemoveVerletIntegrator(GenericOperation):
    """Removes the Verlet integrator operation.

    Parameters
    ----------
    add_op : :class:`AddVerletIntegrator`
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
        The width (in units ``r``) of a bin in the histogram of the rdf.

    """
    def __init__(self, check_thermo_every, check_rdf_every, rdf_dr):
        super().__init__(check_thermo_every, check_rdf_every, rdf_dr)

    def extract_ensemble(self, sim):
        """Creates an ensemble with the averaged thermodynamic properties and rdf.

        Parameters
        ----------
        sim : :class:`~relentless.simulate.simulate.Simulation`
            The simulation object.

        Returns
        -------
        :class:`~relentless.ensemble.Ensemble`
            Ensemble with averaged thermodynamic properties and rdf.

        """
        return self._op.extract_ensemble(sim)
