"""
Simulation interface
====================

Molecular simulation runs are performed in a :class:`Simulation` ensemble container,
which initializes and runs a set of :class:`SimulationOperation`\s. Each simulation
run requires the input of an ensemble, the interaction potentials, and a directory
to write the output data, which are all used to construct a :class:`SimulationInstance`.

The simulations can use a combination of multiple :class:`~relentless.potential.potential.Potential`\s or
:class:`~relentless.potential.pair.PairPotential`\s tabulated together, the interface for
which is given here using :class:`Potentials` and the tabulators :class:`PotentialTabulator`
and :class:`PairPotentialTabulator`.

A number of :class:`Thermostat`\s and :class:`Barostat`\s are also provided to enable
control of the temperature and pressure of the simulation.

.. autosummary::
    :nosignatures:

    Simulation
    SimulationInstance
    Potentials
    PotentialTabulator
    PairPotentialTabulator
    BerendsenThermostat
    NoseHooverThermostat
    BerendsenBarostat
    MTKBarostat

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

To implement your own simulation operation, create a class that derives from
:class:`SimulationOperation` and define the required methods.

To implement your own thermostat or barostat, create a class that derives from
:class:`Thermostat` or :class:`Barostat` and define the required methods.

.. autosummary::
    :nosignatures:

    SimulationOperation
    Thermostat
    Barostat
    GenericOperation

.. autoclass:: Simulation
    :members:
.. autoclass:: SimulationInstance
    :members:
.. autoclass:: SimulationOperation
    :members:
.. autoclass:: Potentials
    :members:
.. autoclass:: PotentialTabulator
    :members:
.. autoclass:: PairPotentialTabulator
    :members:
.. autoclass:: Thermostat
    :members:
.. autoclass:: BerendsenThermostat
    :members:
.. autoclass:: NoseHooverThermostat
    :members:
.. autoclass:: Barostat
    :members:
.. autoclass:: BerendsenBarostat
    :members:
.. autoclass:: MTKBarostat
    :members:

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
import abc
import numpy

from relentless import data

class SimulationOperation(abc.ABC):
    """Abstract base class for operations performed using a simulation engine. To
    implement your own operation, a ``__call__`` method must be defined as specified.
    """
    class Data:
        pass

    @abc.abstractmethod
    def __call__(self, sim):
        pass

class NotImplementedOperation(SimulationOperation):
    def __call__(self, sim):
        raise NotImplementedError('Operation note implemented')

class Simulation:
    """Ensemble simulation container.

    Base class that initializes and runs a simulation described by a set of
    :class:`SimulationOperation`\s.

    Parameters
    ----------
    operations : array_like
        Array of :class:`SimulationOperation`\s to call.
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
        ensemble : :class:`~relentless.ensemble.Ensemble`
            Simulation ensemble. Must include values for ``N`` and ``V`` even if
            these variables fluctuate.
        potentials : :class:`Potentials`
            The interaction potentials.
        directory : str or :class:`~relentless.data.Directory`
            Directory for output.

        Returns
        -------
        :class:`Simulation`
            The simulation instance after the operations are performed.

        Raises
        ------
        TypeError
            If all operations are not :class:`SimulationOperation`\s.

        """
        if not all([isinstance(op,SimulationOperation) for op in self.operations]):
            raise TypeError('All operations must be SimulationOperations.')

        sim = self._new_instance(ensemble, potentials, directory)
        for op in self.operations:
            op(sim)
        return sim

    @property
    def operations(self):
        """list: The operations to be performed during a simulation run."""
        return self._operations

    @operations.setter
    def operations(self, ops):
        try:
            self._operations = list(ops)
        except TypeError:
            self._operations = [ops]

    def _new_instance(self, ensemble, potentials, directory):
        return SimulationInstance(type(self),
                                  ensemble,
                                  potentials,
                                  directory,
                                  **self.options)

    # initialization
    InitializeFromFile = NotImplementedOperation
    InitializeRandomly = NotImplementedOperation

    # energy minimization
    MinimizeEnergy = NotImplementedOperation

    # md integrators
    AddBrownianIntegrator = NotImplementedOperation
    RemoveBrownianIntegrator = NotImplementedOperation
    AddLangevinIntegrator = NotImplementedOperation
    RemoveLangevinIntegrator = NotImplementedOperation
    AddVerletIntegrator = NotImplementedOperation
    RemoveVerletIntegrator = NotImplementedOperation

    # run commands
    Run = NotImplementedOperation
    RunUpTo = NotImplementedOperation

    # analysis
    AddEnsembleAnalyzer = NotImplementedOperation

class SimulationInstance:
    """Specific instance of a simulation and its data.

    Parameters
    ----------
    backend : type
        Type of the simulation class.
    ensemble : :class:`~relentless.ensemble.Ensemble`
        Simulation ensemble. Must include values for ``N`` and ``V`` even if
        these variables fluctuate.
    potentials : :class:`Potentials`
        The interaction potentials.
    directory : str or :class:`~relentless.data.Directory`
        Directory for output.
    options : kwargs
        Optional arguments for the initialize, analyze, and defined "operations" functions.

    """
    def __init__(self, backend, ensemble, potentials, directory, **options):
        self.backend = backend
        self.ensemble = ensemble
        self.potentials = potentials

        if directory is not None:
            directory = data.Directory.cast(directory)
        self.directory = directory

        for opt,val in options.items():
            setattr(self,opt,val)
        self._opdata = {}

    def __getitem__(self, op):
        if not op in self._opdata:
            self._opdata[op] = op.Data()
        return self._opdata[op]

class Potentials:
    """Combination of multiple potentials.

    Initializes a :class:`PairPotentialTabulator` object that can store multiple potentials.
    Before the :class:`Potentials` object can be used, the ``rmax`` and ``num``
    attributes of all ``pair``\s (that are not ``None``) must be set.

    Parameters
    ----------
    pair_potentials : array_like
        The pair potentials to be combined and tabulated. (Defaults to ``None``,
        resulting in an empty :class:`PairPotentialTabulator` object).

    """
    def __init__(self, pair_potentials=None):
        self._pair = PairPotentialTabulator(rmax=None,num=None,neighbor_buffer=0.0,potentials=pair_potentials)

    @property
    def pair(self):
        """:class:`PairPotentialTabulator`: The combined potentials."""
        return self._pair

class PotentialTabulator:
    """Tabulates one or more potentials.

    Enables evaluation of energy, force, and derivative at different
    positional values (i.e. ``x``).

    Parameters
    ----------
    start : float
        The positional value of ``x`` at which to begin tabulation.
    stop : float
        The positional value of ``x`` at which to end tabulation.
    num : int
        The number of points (value of ``x``) at which to tabulate and evaluate the potential.
    potentials : :class:`~relentless.potential.potential.Potential` or array_like
        The potential(s) to be tabulated. If array_like, all elements must
        be :class:`~relentless.potential.potential.Potential`\s. (Defaults to ``None``).

    """
    def __init__(self, start, stop, num, potentials=None):
        self.start = start
        self.stop = stop
        self.num = num
        self.potentials = potentials

    @property
    def potentials(self):
        """array_like: The individual potentials that are tabulated."""
        return self._potentials

    @potentials.setter
    def potentials(self, val):
        if val is not None:
            try:
                self._potentials = list(val)
            except TypeError:
                self._potentials = [val]
        else:
            self._potentials = []

    @property
    def start(self):
        """float: The ``x`` value at which to start tabulation."""
        return self._start

    @start.setter
    def start(self, val):
        self._start = val
        self._compute_x = True

    @property
    def stop(self):
        """float: The ``x`` value at which to stop tabulation."""
        return self._stop

    @stop.setter
    def stop(self, val):
        self._stop = val
        self._compute_x = True

    @property
    def num(self):
        """int: The number of points at which to tabulate/evaluate the potential,
        must be at least 2."""
        return self._num

    @num.setter
    def num(self, val):
        if val is not None and (not isinstance(val,int) or val < 2):
            raise ValueError('Number of points must be at least 2.')
        self._num = val
        self._compute_x = True

    @property
    def x(self):
        """array_like: The values of ``x`` at which to evaluate :meth:`energy`, :meth:`force`, and :meth:`derivative`."""
        if self._compute_x:
            if self.start is None:
                raise ValueError('Start of range must be set.')
            if self.stop is None:
                raise ValueError('End of range must be set.')
            if self.num is None:
                raise ValueError('Number of points must be set.')
            self._x = numpy.linspace(self.start,self.stop,self.num,dtype=numpy.float64)
            self._compute_x = False
        return self._x

    def energy(self, key):
        """Evaluates and accumulates energy for all potentials.

        Parameters
        ----------
        key : str
            The key for which to evaluate the energy for each potential.

        Returns
        -------
        array_like
            Accumulated energy at each ``x`` value.

        """
        u = numpy.zeros_like(self.x)
        for pot in self.potentials:
            try:
                u += pot.energy(key,self.x)
            except KeyError:
                pass
        return u

    def force(self, key):
        """Evaluates and accumulates force for all potentials.

        Parameters
        ----------
        key : str
            The key for which to evaluate the force for each potential.

        Returns
        -------
        array_like
            Accumulated force at each ``x`` value.

        """
        f = numpy.zeros_like(self.x)
        for pot in self.potentials:
            try:
                f += pot.force(key,self.x)
            except KeyError:
                pass
        return f

    def derivative(self, key, var):
        """Evaluates and accumulates derivative for all potentials.

        Parameters
        ----------
        key : str
            The key for which to evaluate the derivative for each potential.

        Returns
        -------
        array_like
            Accumulated force at each ``x`` value.

        """
        d = numpy.zeros_like(self.x)
        for pot in self.potentials:
            try:
                d += pot.derivative(key,var,self.x)
            except KeyError:
                pass
        return d

class PairPotentialTabulator(PotentialTabulator):
    """Tabulates one or more pair potentials.

    Enables evaluation of energy, force, and derivative at different
    positional values (i.e. ``r``).

    Parameters
    ----------
    rmax : float
        The maximum value of ``r`` at which to tabulate.
    num : int
        The number of points (value of ``r``) at which to tabulate and evaluate the potential.
    neighbor_buffer : float
        Buffer radius used in computing the neighbor list.
    potentials : :class:`~relentless.potential.pair.PairPotential` or array_like
        The pair potential(s) to be tabulated. If array_like, all elements must
        be :class:`~relentless.potential.pair.PairPotential`\s. (Defaults to ``None``).
    fmax : float
        The maximum value of force at which to allow evaluation.

    """
    def __init__(self, rmax, num, neighbor_buffer, potentials=None, fmax=None):
        super().__init__(0,rmax,num,potentials)
        self.neighbor_buffer = neighbor_buffer
        self.fmax = fmax

    @property
    def r(self):
        """array_like: The values of ``r`` at which to evaluate :meth:`energy`, :meth:`force`, and :meth:`derivative`."""
        return self.x

    @property
    def rmax(self):
        """float: The maximum value of ``r`` at which to allow tabulation."""
        return self.stop

    @rmax.setter
    def rmax(self, val):
        self.stop = val

    @property
    def neighbor_buffer(self):
        """float: The amount to be added to ``rmax`` to search for particles while
        computing the neighbor list."""
        return self._neighbor_buffer

    @neighbor_buffer.setter
    def neighbor_buffer(self, val):
        self._neighbor_buffer = val

    @property
    def fmax(self):
        """float: The maximum value of force at which to allow evaluation."""
        return self._fmax

    @fmax.setter
    def fmax(self, val):
        if val is not None:
            self._fmax = numpy.fabs(val)
        else:
            self._fmax = None

    def energy(self, pair):
        """Evaluates and accumulates energy for all potentials.

        Shifts the energy to be 0 at ``rmax``.

        Parameters
        ----------
        pair : str
            The pair for which to evaluate the energy for each potential.

        Returns
        -------
        array_like
            Accumulated energy at each ``r`` value.

        """
        u = super().energy(pair)
        u -= u[-1]
        return u

    def force(self, pair):
        """Evaluates and accumulates force for all potentials.

        If set, all forces are truncated to be less than or equal to the `magnitude`
        of ``fmax``.

        Parameters
        ----------
        pair : str
            The pair for which to evaluate the force for each potential.

        Returns
        -------
        array_like
            Accumulated force at each ``r`` value.

        """
        f = super().force(pair)
        if self.fmax is not None:
            flags = numpy.fabs(f) >= self.fmax
            sign = numpy.sign(f[flags])
            f[flags] = sign*self.fmax
        return f

    def derivative(self, pair, var):
        """Evaluates and accumulates derivative for all potentials.

        Shifts the derivative to be 0 at :attr:`rmax`.

        Parameters
        ----------
        pair : str
            The pair for which to evaluate the derivative for each potential.

        Returns
        -------
        array_like
            Accumulated derivative at each ``r`` value.

        """
        d = super().derivative(pair, var)
        d -= d[-1]
        return d

class Thermostat:
    """Generic thermostat.

    Controls the temperature of particles in the simulation by attempting to
    equilibrate the system to the specified temperature.

    Parameters
    ----------
    T : float
        Thermostat target temperature.

    """
    def __init__(self, T):
        self.T = T

class BerendsenThermostat(Thermostat):
    """Berendsen thermostat.

    Parameters
    ----------
    T : float
        Thermostat target temperature.
    tau : float
        Thermostat coupling constant.

    """
    def __init__(self, T, tau):
        super().__init__(T)
        self.tau = tau

class NoseHooverThermostat(Thermostat):
    """Nosé-Hoover thermostat.

    Parameters
    ----------
    T : float
        Thermostat target temperature.
    tau : float
        Thermostat coupling constant.

    """
    def __init__(self, T, tau):
        super().__init__(T)
        self.tau = tau

class Barostat:
    """Generic barostat.

    Controls the pressure of particles in the simulation by attempting to
    equilibrate the system to the specified pressure.

    Parameters
    ----------
    P : float
        Barostat target pressure.

    """
    def __init__(self, P):
        self.P = P

class BerendsenBarostat(Barostat):
    """Berendsen barostat.

    Parameters
    ----------
    P : float
        Barostat target pressure.
    tau : float
        Barostat coupling constant.

    """
    def __init__(self, P, tau):
        super().__init__(P)
        self.tau = tau

class MTKBarostat(Barostat):
    """MTK barostat.

    Parameters
    ----------
    P : float
        Barostat target pressure.
    tau : float
        Barostat coupling constant.

    """
    def __init__(self, P, tau):
        super().__init__(P)
        self.tau = tau

class GenericOperation(SimulationOperation):
    """Generic simulation operation adapter.

    Translates a generic simulation operation into an implemented operation
    for a valid :class:`Simulation` backend. The backend must be an attribute
    of the :class:`GenericOperation`.

    Parameters
    ----------
    args : args
        Positional arguments for simulation operation.
    kwargs : kwargs
        Keyword arguments for simulation operation.

    """
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self._op = None
        self._backend = None

    def __call__(self, sim):
        """Evaluates the generic simulation operation.

        Parameters
        ----------
        sim : :class:`Simulation`
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
            backend = sim.backend
            if not issubclass(backend, Simulation):
                raise TypeError('Backend must be a Simulation.')

            op_name = type(self).__name__
            BackendOp = getattr(backend, op_name, None)
            if BackendOp is NotImplementedOperation or BackendOp is None:
                raise NotImplementedError('{}.{} operation not implemented.'.format(backend.__name__,op_name))

            self._op = BackendOp(*self.args,**self.kwargs)
            self._backend = backend

        return self._op(sim)

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

# run commands
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

# analyzers
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
