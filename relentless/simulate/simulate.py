"""
Simulation interface
====================

See :mod:`relentless.simulate` for module level documentation.

"""
import abc
from enum import Enum
import itertools

import numpy
import scipy.spatial

from relentless import data
from relentless import extent

class SimulationOperation(abc.ABC):
    """Operation to be performed by a :class:`Simulation`."""
    class Data:
        pass

    @abc.abstractmethod
    def __call__(self, sim):
        pass

class NotImplementedOperation(SimulationOperation):
    """Operation not implemented by a :class:`Simulation`."""
    def __call__(self, sim):
        raise NotImplementedError('Operation not implemented')

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

        return self._op(sim)

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

## initializers
class InitializeFromFile(GenericOperation):
    """Initialize a simulation from a file.

    Parameters
    ----------
    filename : str
        The file from which to read the system data.

    """
    def __init__(self, filename):
        super().__init__(filename)

class InitializeRandomly(GenericOperation):
    """Initialize a randomly generated simulation box.

    If ``diameters`` is ``None``, the particles are randomly placed in the box.
    This can work pretty well for low densities, particularly if
    :class:`MinimizeEnergy` is used to remove overlaps before starting to run a
    simulation. However, it will typically fail for higher densities, where
    there are many overlaps that are hard to resolve.

    If ``diameters`` is specified for each particle type, the particles will
    be randomly packed into sites of a close-packed lattice. The insertion
    order is from big to small. No particles are allowed to overlap based on
    the diameters, which typically means the initially state will be more
    favorable than using random initialization. However, the packing procedure
    can fail if there is not enough room in the box to fit particles using
    lattice sites.

    Parameters
    ----------
    seed : int
        The seed to randomly initialize the particle locations.
    N : dict
        Number of particles of each type.
    V : :class:`~relentless.extent.Extent`
        Simulation extent.
    T : float
        Temperature. Defaults to None, which means system is not thermalized.
    masses : dict
        Masses of each particle type. Defaults to None, which means particles
        have unit mass.
    diameters : dict
        Diameter of each particle type. Defaults to None, which means particles
        are randomly inserted without checking their sizes.

    """
    def __init__(self, seed, N, V, T=None, masses=None, diameters=None):
        super().__init__(seed, N, V, T, masses)

    @classmethod
    def _make_orthorhombic(cls, V):
        # get the orthorhombic bounding box
        if isinstance(V, extent.TriclinicBox):
            Lx,Ly,Lz,xy,xz,yz = V.as_array(extent.TriclinicBox.Convention.HOOMD)
            aabb = numpy.array([Lx/numpy.sqrt(1.+xy**2+(xy*yz-xz)**2), Ly/numpy.sqrt(1.+yz**2), Lz])
        elif isinstance(V, extent.ObliqueArea):
            Lx,Ly,xy = V.as_array(extent.ObliqueArea.Convention.HOOMD)
            aabb = numpy.array([Lx/numpy.sqrt(1.+xy**2), Ly])
        else:
            raise TypeError('Random initialization currently only supported in triclinic/oblique extents')
        return aabb

    @classmethod
    def _random_particles(cls, seed, N, V):
        rng = numpy.random.default_rng(seed)
        aabb = cls._make_orthorhombic(V)

        positions = aabb*rng.uniform(size=(sum(N.values()), len(aabb)))
        positions = V.wrap(positions)

        types = []
        for i,Ni in N.items():
            types.extend([i]*Ni)

        return positions, types

    @classmethod
    def _pack_particles(cls, seed, N, V, diameters):
        rng = numpy.random.default_rng(seed)
        aabb = cls._make_orthorhombic(V)
        dimension = len(aabb)
        positions = numpy.zeros((sum(N.values()), dimension), dtype=numpy.float64)
        types = []
        trees = {}
        Nadded = 0
        # insert the particles, big to small
        sorted_N = sorted(N.items(), key=lambda x : x[1], reverse=True)
        for i,Ni in sorted_N:
            # generate site coordinates, on orthorhombic lattices
            di = diameters[i]
            if dimension == 3:
                # fcc lattice
                a = numpy.sqrt(2.)*di
                lattice = a*numpy.array([1.,1.,1.])
                unitcell = a*numpy.array([[0.,0.,0.],[0.5,0.5,0.],[0.5,0.,0.5],[0.,0.5,0.5]])
            elif dimension == 2:
                a = di
                b = numpy.sqrt(3.)*di
                lattice = numpy.array([a,b])
                unitcell = numpy.array([[0.,0.],[0.5*a,0.5*b]])
            else:
                raise ValueError('Only 3d and 2d packings are supported')
            num_lattice = numpy.floor((aabb-di)/lattice).astype(int)
            sites = numpy.zeros((numpy.prod(num_lattice)*unitcell.shape[0], dimension), dtype=numpy.float64)
            first = 0
            for coord in itertools.product(*[numpy.arange(n) for n in num_lattice]):
                sites[first:first+unitcell.shape[0]] = coord*lattice + unitcell
                first += unitcell.shape[0]
            sites += 0.5*di

            # eliminate overlaps using kd-tree collision detection
            if len(trees) > 0:
                mask = numpy.ones(sites.shape[0], dtype=bool)
                for j,treej in trees.items():
                    dj = diameters[j]
                    num_overlap = treej.query_ball_point(sites, 0.5*(di+dj), return_length=True)
                    mask[num_overlap > 0] = False
                sites = sites[mask]

            # randomly draw positions from available sites
            if Ni > sites.shape[0]:
                raise RuntimeError('Failed to randomly pack this box')
            ri = sites[rng.choice(sites.shape[0], Ni, replace=False)]
            # also make tree from positions if we have more than 1 type, using pbcs
            if len(N) > 1:
                trees[i] = scipy.spatial.KDTree(ri)
            positions[Nadded:Nadded+Ni] = ri
            types += [i]*Ni
            Nadded += Ni

        # wrap the particles back into the real box
        positions = V.wrap(positions)

        return positions,types

## integrators
class MinimizeEnergy(GenericOperation):
    """Run an energy minimization until converged.

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
    """Add a Brownian dynamics integration scheme.

    Parameters
    ----------
    dt : float
        Time step size for each simulation iteration
    T : float
        Temperature.
    friction : float
        Sets drag coefficient for each particle type.
    seed : int
        Seed used to randomly generate a uniform force.

    """
    def __init__(self, dt, T, friction, seed):
        super().__init__(dt, T, friction, seed)

class RemoveBrownianIntegrator(GenericOperation):
    """Remove a Brownian dynamics integrator.

    Parameters
    ----------
    add_op : :class:`AddBrownianIntegrator`
        The integrator addition operation to be removed.

    """
    def __init__(self, add_op):
        super().__init__(add_op)

class AddLangevinIntegrator(GenericOperation):
    """Add a Langevin dynamics integration scheme.

    Parameters
    ----------
    dt : float
        Time step size for each simulation iteration
    T : float
        Temperature.
    friction : float or dict
        Sets drag coefficient for each particle type (shared or per-type).
    seed : int
        Seed used to randomly generate a uniform force.

    """
    def __init__(self, dt, T, friction, seed):
        super().__init__(dt, T, friction, seed)

class RemoveLangevinIntegrator(GenericOperation):
    """Remove a Langevin dynamics integrator.

    Parameters
    ----------
    add_op : :class:`AddLangevinIntegrator`
        The integrator addition operation to be removed.

    """
    def __init__(self, add_op):
        super().__init__(add_op)

class AddVerletIntegrator(GenericOperation):
    """Add a Verlet-style integrator.

    The Verlet-style integrator is used to implement classical molecular
    dynamics equations of motion. The integrator may optionally accept a
    :class:`Thermostat` and a :class:`Barostat` for temperature and pressure
    control, respectively. Depending on the :class:`Simulation` engine, not
    all combinations of ``thermostat`` and ``barostat`` may be allowed. Refer
    to the specific documentation for the engine you plan to use if you are
    unsure or obtain an error for your chosen combination.

    .. rubric:: Thermostats
    .. autosummary::
        :nosignatures:

        BerendsenThermostat
        NoseHooverThermostat

    .. rubric:: Barostats
    .. autosummary::
        :nosignatures:

        BerendsenBarostat
        MTKBarostat

    Parameters
    ----------
    dt : float
        Time step size for each simulation iteration.
    thermostat : :class:`~relentless.simulate.simulate.Thermostat`
        Thermostat used for integration (defaults to ``None``).
    barostat : :class:`~relentless.simulate.simulate.Barostat`
        Barostat used for integration (defaults to ``None``).

    """
    def __init__(self, dt, thermostat=None, barostat=None):
        super().__init__(dt, thermostat, barostat)

class RemoveVerletIntegrator(GenericOperation):
    """Remove a Verlet-style integrator.

    Parameters
    ----------
    add_op : :class:`AddVerletIntegrator`
        The integrator addition operation to be removed.

    """
    def __init__(self, add_op):
        super().__init__(add_op)

# run commands
class Run(GenericOperation):
    """Advance the simulation by a given number of time steps.

    Parameters
    ----------
    steps : int
        Number of steps to run.

    """
    def __init__(self, steps):
        super().__init__(steps)

class RunUpTo(GenericOperation):
    """Advance the simulation up to a given time step.

    Parameters
    ----------
    step : int
        Step number up to which to run.

    """
    def __init__(self, step):
        super().__init__(step)

# analyzers
class AddEnsembleAnalyzer(GenericOperation):
    """Analyze the simulation ensemble.

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
        """Create an ensemble with the averaged thermodynamic properties and rdf.

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
