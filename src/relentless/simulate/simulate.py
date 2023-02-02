"""
Simulation interface
====================

See :mod:`relentless.simulate` for module level documentation.

"""
import abc

import numpy

from relentless import data, mpi


class SimulationOperation(abc.ABC):
    """Operation to be performed by a :class:`Simulation`."""

    @abc.abstractmethod
    def __call__(self, sim):
        pass


class InitializationOperation(SimulationOperation):
    """Operation that initializes a :class:`Simulation`.

    An initialization operation must do the following:

    * Set `sim.dimension`, the dimensionality of the simulation.
    * Set `sim.types`, the list of particle types in the simulation.
    * Attach the :class:`Potentials` given by `sim.potentials`.

    The initialization operation may additionally stash any data required to
    run the simulation (e.g., a specific instance of the simulation engine).
    These should be stored as private data keys beginning with an `_`.

    """

    pass


class AnalysisOperation(abc.ABC):
    """Analysis operation to be performed by a :class:`Simulation`."""

    @abc.abstractmethod
    def pre_run(self, sim, sim_op):
        pass

    @abc.abstractmethod
    def post_run(self, sim, sim_op):
        pass

    @abc.abstractmethod
    def process(self, sim, sim_op):
        pass


class DelegatedSimulationOperation(SimulationOperation):
    def __call__(self, sim):
        # run operation
        op = self._make_delegate(sim)
        op(sim)
        # pilfer data and delete
        sim[self] = dict(sim[op])
        del sim[op]
        del op

    def _get_delegate(self, sim, *args, **kwargs):
        op_name = type(self).__name__
        delegate = getattr(sim.backend, "_" + op_name, None)
        if delegate is None:
            raise NotImplementedError(
                "{} is not implemented for {}.".format(op_name, sim.backend.__name__)
            )
        return delegate(*args, **kwargs)

    @abc.abstractmethod
    def _make_delegate(self, sim):
        pass


class DelegatedInitializationOperation(
    DelegatedSimulationOperation, InitializationOperation
):
    pass


class DelegatedAnalysisOperation(AnalysisOperation):
    def pre_run(self, sim, sim_op):
        delegate_op = self._make_delegate(sim)
        delegate_op.pre_run(sim, sim_op)

        sim[self] = dict(sim[delegate_op])
        sim[self].update({"_delegate": delegate_op, "_delegate_state": "pre_run"})

    def post_run(self, sim, sim_op):
        if "_delegate" not in sim[self] or sim[self]["_delegate_state"] != "pre_run":
            raise RuntimeError("Call pre_run before post_run")
        delegate_op = sim[self]["_delegate"]
        delegate_op.post_run(sim, sim_op)

        sim[self].update(sim[delegate_op])
        sim[self]["_delegate_state"] = "post_run"

    def process(self, sim, sim_op):
        if "_delegate" not in sim[self] or sim[self]["_delegate_state"] != "post_run":
            raise RuntimeError("Call post_run before process")
        delegate_op = sim[self]["_delegate"]
        delegate_op.process(sim, sim_op)

        sim[self].update(sim[delegate_op])

        del sim[delegate_op]
        del sim[self]["_delegate"]
        del sim[self]["_delegate_state"]

    _get_delegate = DelegatedSimulationOperation._get_delegate

    @abc.abstractmethod
    def _make_delegate(self, sim):
        pass


class Simulation:
    r"""Simulation engine.

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
        r"""Run the simulation operations.

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
        sim = self._new_instance(potentials, directory)
        # run the operations
        for op in self.operations:
            op(sim)
        # execute post run commands
        self._post_run(sim)

        return sim

    def _post_run(self, sim):
        # finalize the analysis operations
        for op in self.operations:
            if hasattr(op, "analyzers"):
                for analyzer in op.analyzers:
                    analyzer.process(sim, op)

    @property
    def initializer(self):
        """:class:`SimulationOperation`: Initialization operation."""
        return self._initializer

    @initializer.setter
    def initializer(self, op):
        if not isinstance(op, InitializationOperation):
            return TypeError("Initializer must be an InitializationOperation")
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
            raise TypeError("All operations must be SimulationOperations.")
        self._operations = ops_

    def _new_instance(self, potentials, directory):
        sim = SimulationInstance(type(self), self.initializer, potentials, directory)
        sim.initializer(sim)
        return sim


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

        self.directory = data.Directory.cast(directory, create=mpi.world.rank_is_root)
        mpi.world.barrier()

        # properties of simulation, to be set
        self.dimension = None
        self._types = None
        self._pairs = None

        # operation data set
        self._opdata = {}

        self.initializer = initializer

    def __getitem__(self, op):
        if op not in self._opdata:
            self._opdata[op] = {}
        return self._opdata[op]

    def __setitem__(self, op, value):
        self._opdata[op] = {}
        self._opdata[op].update(value)

    def __delitem__(self, op):
        del self._opdata[op]

    @property
    def types(self):
        """tuple: Particle types in simulation."""
        return self._types

    @types.setter
    def types(self, value):
        self._types = tuple(value)
        self._pairs = tuple((i, j) for i in self._types for j in self._types if j >= i)

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
        self._pair = PairPotentialTabulator(
            start=0.0,
            stop=None,
            num=None,
            neighbor_buffer=0.0,
            potentials=pair_potentials,
        )
        self.kB = kB

    @property
    def pair(self):
        """:class:`PairPotentialTabulator`: Pair potentials."""
        return self._pair


class PotentialTabulator:
    r"""Tabulator for an interaction potential.

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
        The number of points (value of ``x``) at which to tabulate and evaluate
        the potential.
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

    @property
    def stop(self):
        """float: The ``x`` value at which to stop tabulation."""
        return self._stop

    @stop.setter
    def stop(self, val):
        self._stop = val

    @property
    def num(self):
        """int: The number of points at which to tabulate/evaluate the potential,
        must be at least 2."""
        return self._num

    @num.setter
    def num(self, val):
        if val is not None and (not isinstance(val, int) or val < 2):
            raise ValueError("Number of points must be at least 2.")
        self._num = val

    def validate(self):
        if self.start is None:
            raise ValueError("Start of range must be set.")
        if self.stop is None:
            raise ValueError("End of range must be set.")
        if self.num is None:
            raise ValueError("Number of points must be set.")
        if self.start >= self.stop:
            raise ValueError("Range must be increasing")

    @property
    def x(self):
        """array_like: The values of ``x`` at which to evaluate the potential."""
        self.validate()
        return numpy.linspace(self.start, self.stop, self.num, dtype=float)

    @property
    def xsquared(self):
        """array_like: The values of ``x**2`` at which to evaluate the potential."""
        self.validate()
        return numpy.linspace(self.start**2, self.stop**2, self.num, dtype=float)

    def energy(self, key, x=None):
        """Evaluates and accumulates energy for all potentials.

        Parameters
        ----------
        key : str
            The key for which to evaluate the energy for each potential.
        x : float or list
            The pair distance(s) at which to evaluate the derivative.
            Default of ``None`` will use a linear space from `start` to `stop`.

        Returns
        -------
        array_like
            Accumulated energy at each ``x`` value.

        """
        if x is None:
            x = self.x
        u = numpy.zeros_like(x, dtype=float)
        for pot in self.potentials:
            try:
                u += pot.energy(key, x)
            except KeyError:
                pass
        return u

    def force(self, key, x=None):
        """Evaluates and accumulates force for all potentials.

        Parameters
        ----------
        key : str
            The key for which to evaluate the force for each potential.
        x : float or list
            The pair distance(s) at which to evaluate the derivative.
            Default of ``None`` will use a linear space from `start` to `stop`.

        Returns
        -------
        array_like
            Accumulated force at each ``x`` value.

        """
        if x is None:
            x = self.x
        f = numpy.zeros_like(x, dtype=float)
        for pot in self.potentials:
            try:
                f += pot.force(key, x)
            except KeyError:
                pass
        return f

    def derivative(self, key, var, x=None):
        """Evaluates and accumulates derivative for all potentials.

        Parameters
        ----------
        key : str
            The key for which to evaluate the derivative for each potential.
        var : :class:`~relentless.model.Variable`
            Variable to differentiate with respect to.
        x : float or list
            The pair distance(s) at which to evaluate the derivative.
            Default of ``None`` will use a linear space from `start` to `stop`.

        Returns
        -------
        array_like
            Accumulated force at each ``x`` value.

        """
        if x is None:
            x = self.x
        d = numpy.zeros_like(x, dtype=float)
        for pot in self.potentials:
            try:
                d += pot.derivative(key, var, x)
            except KeyError:
                pass
        return d


class PairPotentialTabulator(PotentialTabulator):
    r"""Tabulate one or more pair potentials.

    Enables evaluation of energy, force, and derivative at different
    positional values (i.e. ``r``).

    Parameters
    ----------
    start : float
        The minimum value of ``r`` at which to tabulate.
    stop : float
        The maximum value of ``r`` at which to tabulate.
    num : int
        The number of points (value of ``r``) at which to tabulate and evaluate the
        potential.
    neighbor_buffer : float
        Buffer radius used in computing the neighbor list.
    potentials : :class:`~relentless.potential.pair.PairPotential` or array_like
        The pair potential(s) to be tabulated. If array_like, all elements must
        be :class:`~relentless.potential.pair.PairPotential`\s. (Defaults to ``None``).
    fmax : float
        The maximum value of force at which to allow evaluation.

    """

    def __init__(self, start, stop, num, neighbor_buffer, potentials=None, fmax=None):
        super().__init__(start, stop, num, potentials)
        self.neighbor_buffer = neighbor_buffer
        self.fmax = fmax

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

    def energy(self, pair, x=None):
        """Evaluates and accumulates energy for all potentials.

        Shifts the energy to be 0 at `stop`.

        Parameters
        ----------
        pair : str
            The pair for which to evaluate the energy for each potential.
        x : float or list
            The pair distance(s) at which to evaluate the derivative.
            Default of ``None`` will use a linear space from `start` to `stop`.

        Returns
        -------
        array_like
            Accumulated energy at each ``r`` value.

        """
        u = super().energy(pair, x) - super().energy(pair, self.stop)
        return u

    def force(self, pair, x=None):
        """Evaluates and accumulates force for all potentials.

        If set, all forces are truncated to be less than or equal to the `fmax`
        of ``fmax``.

        Parameters
        ----------
        pair : str
            The pair for which to evaluate the force for each potential.
        x : float or list
            The pair distance(s) at which to evaluate the derivative.
            Default of ``None`` will use a linear space from `start` to `stop`.

        Returns
        -------
        array_like
            Accumulated force at each ``r`` value.

        """
        f = super().force(pair, x)
        if self.fmax is not None:
            flags = numpy.fabs(f) >= self.fmax
            sign = numpy.sign(f[flags])
            f[flags] = sign * self.fmax
        return f

    def derivative(self, pair, var, x=None):
        """Evaluates and accumulates derivative for all potentials.

        Shifts the derivative to be 0 at :attr:`rmax`.

        Parameters
        ----------
        pair : str
            The pair for which to evaluate the derivative for each potential.
        var : :class:`~relentless.model.Variable`
            Variable to differentiate with respect to.
        x : float or list
            The pair distance(s) at which to evaluate the derivative.
            Default of ``None`` will use a linear space from `start` to `stop`.

        Returns
        -------
        array_like
            Accumulated derivative at each ``r`` value.

        """
        d = super().derivative(pair, var, x) - super().derivative(pair, var, self.stop)
        return d
