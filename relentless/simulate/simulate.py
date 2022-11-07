"""
Simulation interface
====================

See :mod:`relentless.simulate` for module level documentation.

"""
import abc

import numpy

from relentless import data

class SimulationOperation(abc.ABC):
    """Operation to be performed by a :class:`Simulation`."""
    class Data:
        pass

    @abc.abstractmethod
    def __call__(self, sim):
        pass

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
        self._op = None
        self._backend = None
        self._args = args
        self._kwargs = kwargs
        self._forward_attr = set()

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
        self._ensure_op(sim)
        return self._op(sim)

    def _ensure_op(self, sim):
        if self._op is None or self._backend != sim.backend:
            backend = sim.backend
            if not issubclass(backend, Simulation):
                raise TypeError('Backend must be a Simulation.')

            op_name = type(self).__name__
            BackendOp = getattr(backend, op_name, None)
            if BackendOp is NotImplementedOperation or BackendOp is None:
                raise NotImplementedError('{}.{} operation not implemented.'.format(backend.__name__,op_name))

            self._op = BackendOp(*self._args,**self._kwargs)
            for attr in self._forward_attr:
                value = getattr(self, attr)
                setattr(self._op, attr, value)
            self._backend = backend

    def __getattr__(self, name):
        if self._op is not None:
            if not hasattr(self._op, name):
                raise AttributeError('Operation does not have attribute')
            return getattr(self._op, name)
        else:
            raise AttributeError('Backend operation not initialized yet')

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name not in ('_op','_backend','_args','_kwargs','_forward_attr'):
            self._forward_attr.add(name)
            if self._op is not None:
                setattr(self._op, name, value)

class NotImplementedOperation(SimulationOperation):
    """Operation not implemented by a :class:`Simulation`."""
    def __call__(self, sim):
        raise NotImplementedError('Operation not implemented')

class AnalysisOperation(SimulationOperation):
    @abc.abstractmethod
    def finalize(self, sim):
        pass

class GenericAnalysisOperation(AnalysisOperation, GenericOperation):
    def finalize(self, sim):
        self._ensure_op(sim)
        self._op.finalize(sim)

class Simulation:
    """Simulation engine.

    A simulation engine interprets a sequence of :class:`SimulationOperation`\s
    into a set of commands or parameters that are executed by backend software.
    The easiest backend software to work with will have a Python interface so
    that the operation commands can be passed through to Python commands.
    However, in-house or legacy software may require creation of input files,
    etc. that are executed by other binaries or programs.

    When the simulation is :meth:`run`, a :class:`SimulationInstance` is created
    that can hold specific data or parameters associated with the operations.
    Hence, a single :class:`Simulation` can be run multiple times, e.g., with
    varied inputs, to obtain different output :class:`SimulationInstance`\s.

    Parameters
    ----------
    initializer : :class:`SimulationOperation`
        Operation to initialize the simulation.
    operations : array_like
        Sequence of :class:`SimulationOperation`\s to call.

    """
    def __init__(self, initializer, operations=None):
        self.initializer = initializer
        if operations is not None:
            self.operations = operations
        else:
            self.operations = []

    def run(self, potentials, directory):
        """Run the simulation operations.

        A new simulation instance is created to perform the run. It is intended
        to be destroyed at the end of the run to prevent memory leaks.

        Parameters
        ----------
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
        # initialize the instance
        sim = self._new_instance(self.initializer, potentials, directory)
        # then run the remaining operations
        for op in self.operations:
            if isinstance(op, AnalysisOperation):
                raise TypeError('Analysis operations should be attached to another operation')
            op(sim)
        return sim

    @property
    def initializer(self):
        """:class:`SimulationOperation`: Initialization operation."""
        return self._initializer

    @initializer.setter
    def initializer(self, op):
        if not isinstance(op, SimulationOperation):
            return TypeError('Initializer must be SimulationOperation')
        self._initializer = op

    @property
    def operations(self):
        """list: The operations to be performed during a simulation run."""
        return self._operations

    @operations.setter
    def operations(self, ops):
        try:
            ops_ = list(ops)
        except TypeError:
            ops_ = [ops]
        if not all([isinstance(op, SimulationOperation) for op in ops_]):
            raise TypeError('All operations must be SimulationOperations.')
        self._operations = ops_

    def _new_instance(self, initializer, potentials, directory):
        return SimulationInstance(type(self),
                                  initializer,
                                  potentials,
                                  directory)

    # initialization
    InitializeFromFile = NotImplementedOperation
    InitializeRandomly = NotImplementedOperation

    # energy minimization
    MinimizeEnergy = NotImplementedOperation

    # md integrators
    RunBrownianDynamics = NotImplementedOperation
    RunLangevinDynamics = NotImplementedOperation
    RunMolecularDynamics = NotImplementedOperation

    # analysis
    EnsembleAverage = NotImplementedOperation

class SimulationInstance:
    """Specific instance of a simulation and its data.

    Parameters
    ----------
    backend : type
        Type of the simulation class.
    initializer : :class:`SimulationOperation`
        Operation to initialize the simulation.
    potentials : :class:`Potentials`
        The interaction potentials.
    directory : str or :class:`~relentless.data.Directory`
        Directory for output.

    Attributes
    ----------
    backend : type
        Backend class type for the simulation.
    potentials : :class:`Potentials`
        Potentials for the simulation.
    directory : :class:`~relentless.data.Directory`
        Directory for simulation data.    
    dimension : int
        Dimensionality of the simulation.
    initializer : :class:`SimulationOperation`
        Operation to initialize the simulation.

    """
    def __init__(self, backend, initializer, potentials, directory):
        self.backend = backend
        self.potentials = potentials

        if directory is not None:
            directory = data.Directory.cast(directory)
        self.directory = directory

        # properties of simulation, to be set
        self.dimension = None
        self._types = None
        self._pairs = None

        # operation data set
        self._opdata = {}

        # finish running the setup with the initializer
        if not isinstance(initializer, SimulationOperation):
            raise TypeError('Initializer must be a SimulationOperation')
        self.initializer = initializer

    def __getitem__(self, op):
        op_ = op._op if isinstance(op, GenericOperation) else op
        if not op_ in self._opdata:
            self._opdata[op_] = op_.Data()
        return self._opdata[op_]

    @property
    def types(self):
        """tuple: Particle types in simulation."""
        return self._types

    @types.setter
    def types(self, value):
        self._types = tuple(value)
        self._pairs = tuple((i,j) for i in self._types for j in self._types if j >= i)

    @property
    def pairs(self):
        """tuple: Unique pairs of particle types in simulation."""
        return self._pairs

class Potentials:
    """Set of interaction potentials.

    This class combines one or more potentials to be used as input to
    :meth:`Simulation.run`. In order to enable a common interface, potentials
    must be organized by type (e.g., pair potentials). Potentials of a common
    type are then numerically tabulated, as many simulation packages can
    accommodate tabulated inputs with complex parametrization.

    The parameters of these tabulators, which are automatically initialized,
    must be specified before a simulation can be run.

    Parameters
    ----------
    pair_potentials : array_like
        The pair potentials to be combined and tabulated. (Defaults to ``None``,
        resulting in an empty :class:`PairPotentialTabulator` object).
    kB : float
        Boltzmann constant in your units.

    Attributes
    ----------
    kB : float
        Boltzmann constant.

    """
    def __init__(self, pair_potentials=None, kB=1.0):
        self._pair = PairPotentialTabulator(rmin=0.0,
                                            rmax=None,
                                            num=None,
                                            neighbor_buffer=0.0,
                                            potentials=pair_potentials)
        self.kB = kB

    @property
    def pair(self):
        """:class:`PairPotentialTabulator`: Pair potentials."""
        return self._pair

class PotentialTabulator:
    """Tabulator for an interaction potential.

    The energy, force, and derivative are evaluated at different positions ``x``
    and stored in arrays. The ``start`` and ``end`` of this range must be
    specified, along with the number of points (``num``) to use.

    If no ``potentials`` are specified, the tabulator returns zeros for all.

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
    """Tabulate one or more pair potentials.

    Enables evaluation of energy, force, and derivative at different
    positional values (i.e. ``r``).

    Parameters
    ----------
    rmin : float
        The minimum value of ``r`` at which to tabulate.
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
    def __init__(self, rmin, rmax, num, neighbor_buffer, potentials=None, fmax=None):
        super().__init__(rmin,rmax,num,potentials)
        self.neighbor_buffer = neighbor_buffer
        self.fmax = fmax

    @property
    def r(self):
        """array_like: The values of ``r`` at which to evaluate :meth:`energy`, :meth:`force`, and :meth:`derivative`."""
        return self.x

    @property
    def rmin(self):
        """float: The minimum value of ``r`` at which to allow tabulation."""
        return self.start

    @rmin.setter
    def rmin(self, val):
        self.start = val

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
