"""
HOOMD
=====

This module implements the :class:`HOOMD` simulation backend and its specific
operations. It is best to interface with these operations using the frontend in
:mod:`relentless.simulate`.

.. rubric:: Developer notes

To implement your own HOOMD operation, create an operation that derives from
:class:`~relentless.simulate.simulate.SimulationOperation` and define the
required methods.

.. autoclass:: Initialize
    :members:

"""
import abc
import os
from packaging import version

import numpy

from relentless.collections import FixedKeyDict, PairMatrix
from relentless import ensemble
from relentless import extent
from relentless.math import Interpolator
from relentless import mpi
from . import simulate

try:
    import hoomd
    import hoomd.md
    _hoomd_found = True
except ImportError:
    _hoomd_found = False

try:
    import freud
    _freud_found = True
except ImportError:
    _freud_found = False

## initializers
class Initialize(simulate.SimulationOperation):
    """Initialize a simulation.
    
    This is an abstract base class that needs to have its :meth:`initialize`
    method implemented.
    
    """
    def __call__(self, sim):
        sim[self].system = self.initialize(sim)

    @abc.abstractmethod
    def initialize(self, sim):
        """Initialize the simulation.
        
        Parameters
        ----------
        sim : :class:`~relentless.simulate.SimulationInstance`
            Simulation instance.

        Returns
        -------
        :class:`hoomd.data.system_data`
            HOOMD system.
        
        """
        pass

class InitializeFromFile(Initialize):
    """Initialize a simulation from a GSD file.

    Parameters
    ----------
    filename : str
        The file from which to read the system data.

    """
    def __init__(self, filename):
        super().__init__()
        self.filename = os.path.realpath(filename)

    def initialize(self, sim):
        with sim.hoomd:
            return hoomd.init.read_gsd(self.filename)

class InitializeRandomly(Initialize):
    """Initialize a randomly generated simulation box.

    Particles are randomly placed in the box according to the specification.
    No checks are performed for overlaps, so this can produce very poor starting
    states for denser systems.

    Parameters
    ----------
    seed : int
        The seed to randomly initialize the particle locations.
    N : dict
        Number of particles of each type.
    V : :class:`~relentless.extent.TriclinicBox` or :class:`~relentless.extent.ObliqueArea`
        Simulation extent.
    T : float
        Temperature.
    masses : dict
        Mass of each particle type.

    """
    def __init__(self, seed, N, V, T=None, masses=None):
        super().__init__()
        self.seed = seed
        self.N = N
        self.V = V
        self.T = T
        self.masses = masses

    def initialize(self, sim):
        if not isinstance(self.V, (extent.TriclinicBox, extent.ObliqueArea)):
            raise TypeError('HOOMD boxes must be derived from TriclinicBox or ObliqueArea')

        # if setting seed, preserve the current RNG state
        if self.seed is not None:
            old_state = numpy.random.get_state()
            numpy.random.seed(self.seed)
        else:
            old_state = None

        try:
            # randomly place the particles
            with sim.hoomd:
                # make the box and snapshot
                if isinstance(self.V, extent.TriclinicBox):
                    box_ = hoomd.data.boxdim(Lx=self.V.a[0],
                                             Ly=self.V.b[1],
                                             Lz=self.V.c[2],
                                             xy=self.V.b[0]/self.V.b[1],
                                             xz=self.V.c[0]/self.V.c[2],
                                             yz=self.V.c[1]/self.V.c[2],
                                             dimensions=3)
                    box = freud.Box.from_box([box_.Lx, box_.Ly, box_.Lz, box_.xy, box_.xz, box_.yz], dimensions=3)
                elif isinstance(self.V, extent.ObliqueArea):
                    box_ = hoomd.data.boxdim(Lx=self.V.a[0],
                                             Ly=self.V.b[1],
                                             Lz=1,
                                             xy=self.V.b[0]/self.V.b[1],
                                             xz=0,
                                             yz=0,
                                             dimensions=2)
                    box = freud.Box.from_box([box_.Lx, box_.Ly, 0, box_.xy, 0, 0], dimensions=2)
                else:
                    raise ValueError('HOOMD supports 2d and 3d simulations')

                types = tuple(self.N.keys())
                snap = hoomd.data.make_snapshot(N=sum(self.N.values()),
                                                box=box_,
                                                particle_types=list(types))

                # randomly place particles in fractional coordinates
                if mpi.world.rank == 0:
                    # draw positions
                    rs = numpy.zeros((snap.particles.N, 3))
                    rs[:,:box.dimensions] = numpy.random.uniform(size=(snap.particles.N, box.dimensions))
                    snap.particles.position[:] = box.make_absolute(rs)

                    # set types of each
                    snap.particles.typeid[:] = numpy.repeat(numpy.arange(len(types)), tuple(self.N.values()))

                    # set masses, defaulting to unit mass
                    if self.masses is not None:
                        snap.particles.mass[:] = numpy.repeat([self.masses[t] for t in types], tuple(self.N.values()))
                    else:
                        snap.particles.mass[:] = 1.0

                    # set velocities, defaulting to zeros
                    vel = numpy.zeros((snap.particles.N, 3))
                    if self.T is not None:
                        # Maxwell-Boltzmann = normal with variance kT/m per component
                        v_mb = numpy.random.normal(scale=numpy.sqrt(sim.potentials.kB*self.T),
                                                    size=(snap.particles.N,box.dimensions))
                        v_mb /= numpy.sqrt(snap.particles.mass[:,None])

                        # zero the linear momentum
                        p_mb = numpy.sum(snap.particles.mass[:,None]*v_mb, axis=0)
                        v_cm = p_mb/numpy.sum(snap.particles.mass)
                        v_mb -= v_cm

                        vel[:,:box.dimensions] = v_mb
                    snap.particles.velocity[:] = vel

                # read snapshot
                return hoomd.init.read_snapshot(snap)
        finally:
            # always restore old state if it exists
            if old_state is not None:
                numpy.random.set_state(old_state)

## integrators
class MinimizeEnergy(simulate.SimulationOperation):
    """Run energy minimization until convergence.

    Valid **options** include:

    - **max_displacement** (`float`) - the maximum time step size the minimizer
      is allowed to use.
    - **steps_per_iteration** (`int`) - the number of steps the minimizer runs
      per iteration. Defaults to 100.

    Parameters
    ----------
    energy_tolerance : float
        Energy convergence criterion.
    force_tolerance : float
        Force convergence criterion.
    max_iterations : int
        Maximum number of iterations to run the minimization.
    options : dict
        Additional options for energy minimizer.

    Raises
    ------
    KeyError
        If a value for the maximum displacement is not provided.
    RuntimeError
        If energy minimization has failed to converge within the maximum
        number of iterations.

    """
    def __init__(self, energy_tolerance, force_tolerance, max_iterations, options):
        self.energy_tolerance = energy_tolerance
        self.force_tolerance = force_tolerance
        self.max_iterations = max_iterations
        self.options = options
        if 'max_displacement' not in self.options:
           raise KeyError('HOOMD energy minimizer requires max_displacement option.')
        if 'steps_per_iteration' not in self.options:
            self.options['steps_per_iteration'] = 100

    def __call__(self, sim):
        with sim.hoomd:
            # setup FIRE minimization
            fire = hoomd.md.integrate.mode_minimize_fire(dt=self.options['max_displacement'],
                                                         Etol=self.energy_tolerance,
                                                         ftol=self.force_tolerance)
            all_ = hoomd.group.all()
            nve = hoomd.md.integrate.nve(all_)

            # run while not yet converged
            it = 0
            while not fire.has_converged() and it < self.max_iterations:
                hoomd.run(self.options['steps_per_iteration'])
                it += 1
            if not fire.has_converged():
                raise RuntimeError('Energy minimization failed to converge.')

            # try to cleanup these integrators from the system
            # we want them to ideally be isolated to this method
            nve.disable()
            del nve
            del fire

class AddMDIntegrator(simulate.SimulationOperation):
    """Adds an integrator (for equations of motion) to the simulation.

    Parameters
    ----------
    dt : float
        Time step size for each simulation iteration.

    """
    def __init__(self, dt):
        self.dt = dt

    def attach_integrator(self, sim):
        # note that this assumes you can only have ONE integrator in the system
        #
        # to support multiple methods, you would need to only attach this only if
        # the integrator was not set on the SimulationInstance already
        #
        # doing it this way now because the integration methods all work on all().
        with sim.hoomd:
            hoomd.md.integrate.mode_standard(self.dt)

class RemoveMDIntegrator(simulate.SimulationOperation):
    """Removes a specified integration operation.

    Parameters
    ----------
    add_op : :class:`~relentless.simulate.simulate.SimulationOperation`
        The addition/integration operation to be removed.

    Raises
    ------
    AttributeError
        If the specified integrator has already been removed.

    """
    def __init__(self, add_op):
        self.add_op = add_op

    def __call__(self, sim):
        if sim[self.add_op].integrator is not None:
            with sim.hoomd:
                sim[self.add_op].integrator.disable()
                sim[self.add_op].integrator = None
        else:
            raise AttributeError('The specified integrator has already been removed.')

class AddBrownianIntegrator(AddMDIntegrator):
    """Brownian dynamics for a NVT ensemble.

    Parameters
    ----------
    dt : float
        Time step size for each simulation iteration.
    T : float
        Temperature.
    friction : float
        Sets drag coefficient for each particle type.
    seed : int
        Seed used to randomly generate a uniform force.

    """
    def __init__(self, dt, T, friction, seed):
        super().__init__(dt)
        self.T = T
        self.friction = friction
        self.seed = seed

    def __call__(self, sim):
        self.attach_integrator(sim)
        with sim.hoomd:
            all_ = hoomd.group.all()
            sim[self].integrator = hoomd.md.integrate.brownian(group=all_,
                                                               kT=sim.potentials.kB*self.T,
                                                               seed=self.seed)
            for t in sim.types:
                try:
                    gamma = self.friction[t]
                except TypeError:
                    gamma = self.friction
                sim[self].integrator.set_gamma(t,gamma)

class RemoveBrownianIntegrator(RemoveMDIntegrator):
    """Removes the Brownian integrator operation.

    Parameters
    ----------
    add_op : :class:`AddBrownianIntegrator`
        The integrator addition operation to be removed.

    Raises
    ------
    TypeError
        If the specified addition operation is not a Brownian integrator.

    """
    def __init__(self, add_op):
        super().__init__(add_op)

class AddLangevinIntegrator(AddMDIntegrator):
    """Add a Langevin dynamics integrator.

    Parameters
    ----------
    dt : float
        Time step.
    T : float
        Temperature.
    friction : float or dict
        Drag coefficient for each particle type (shared or per-type).
    seed : int
        Seed used to randomly generate a uniform force.

    """
    def __init__(self, dt, T, friction, seed):
        super().__init__(dt)
        self.T = T
        self.friction = friction
        self.seed = seed

    def __call__(self, sim):
        self.attach_integrator(sim)
        with sim.hoomd:
            all_ = hoomd.group.all()
            sim[self].integrator = hoomd.md.integrate.langevin(group=all_,
                                                               kT=sim.potentials.kB*self.T,
                                                               seed=self.seed)
            for t in sim.types:
                try:
                    gamma = self.friction[t]
                except TypeError:
                    gamma = self.friction
                sim[self].integrator.set_gamma(t,gamma)

class RemoveLangevinIntegrator(RemoveMDIntegrator):
    """Remove a Langevin integrator.

    Parameters
    ----------
    add_op : :class:`AddLangevinIntegrator`
        The add operation for the integrator to remove.

    """
    def __init__(self, add_op):
        super().__init__(add_op)

class AddVerletIntegrator(AddMDIntegrator):
    """Add a Verlet integrator.

    This method supports:

    - NVE integration
    - NVT integration with Nosé-Hoover or Berendsen thermostat
    - NPH integration with MTK barostat
    - NPT integration with Nosé-Hoover thermostat and MTK barostat

    Parameters
    ----------
    dt : float
        Time step.
    thermostat : :class:`~relentless.simulate.simulate.Thermostat`
        Thermostat for temperature control (defaults to ``None``).
    barostat : :class:`~relentless.simulate.simulate.Barostat`
        Barostat for pressure control (defaults to ``None``).

    Raises
    ------
    TypeError
        If an appropriate combination of thermostat and barostat is not set.

    """
    def __init__(self, dt, thermostat=None, barostat=None):
        super().__init__(dt)
        self.thermostat = thermostat
        self.barostat = barostat

    def __call__(self, sim):
        self.attach_integrator(sim)
        with sim.hoomd:
            all_ = hoomd.group.all()
            if self.thermostat is None and self.barostat is None:
                sim[self].integrator = hoomd.md.integrate.nve(group=all_)
            elif isinstance(self.thermostat, simulate.BerendsenThermostat) and self.barostat is None:
                sim[self].integrator = hoomd.md.integrate.berendsen(group=all_,
                                                                    kT=sim.potentials.kB*self.thermostat.T,
                                                                    tau=self.thermostat.tau)
            elif isinstance(self.thermostat, simulate.NoseHooverThermostat) and self.barostat is None:
                sim[self].integrator = hoomd.md.integrate.nvt(group=all_,
                                                              kT=sim.potentials.kB*self.thermostat.T,
                                                              tau=self.thermostat.tau)
            elif self.thermostat is None and isinstance(self.barostat, simulate.MTKBarostat):
                sim[self].integrator = hoomd.md.integrate.nph(group=all_,
                                                              P=self.barostat.P,
                                                              tauP=self.barostat.tau)
            elif isinstance(self.thermostat, simulate.NoseHooverThermostat) and isinstance(self.barostat, simulate.MTKBarostat):
                sim[self].integrator = hoomd.md.integrate.npt(group=all_,
                                                              kT=sim.potentials.kB*self.thermostat.T,
                                                              tau=self.thermostat.tau,
                                                              P=self.barostat.P,
                                                              tauP=self.barostat.tau)
            else:
                raise TypeError('An appropriate combination of thermostat and barostat must be set.')


class RemoveVerletIntegrator(RemoveMDIntegrator):
    """Remove a Verlet integrator.

    Parameters
    ----------
    add_op : :class:`AddVerletIntegrator`
        The add operation for the integrator to remove.

    """
    def __init__(self, add_op):
        super().__init__(add_op)

class Run(simulate.SimulationOperation):
    """Advance the simulation by a given number of timesteps.

    Parameters
    ----------
    steps : int
        Number of steps to run.

    """
    def __init__(self, steps):
        self.steps = steps

    def __call__(self, sim):
        with sim.hoomd:
            hoomd.run(self.steps)

class RunUpTo(simulate.SimulationOperation):
    """Advance the simulation up to a given timestep.

    Parameters
    ----------
    step : int
        Step number up to which to run.

    """
    def __init__(self, step):
        self.step = step

    def __call__(self, sim):
        with sim.hoomd:
            hoomd.run_upto(self.step)

## analyzers
class AddEnsembleAnalyzer(simulate.SimulationOperation):
    """Analyzer for the simulation ensemble.

    The simulation ensemble is analyzed online while it is running. The
    instantaneous thermodynamic properties are extracted from a HOOMD logger,
    while the radial distribution function is computed using :mod:`freud`.

    Parameters
    ----------
    check_thermo_every : int
        Number of timesteps between computing thermodynamic properties.
    check_rdf_every : int
        Number of time steps between computing the RDF.
    rdf_dr : float
        The width of a bin in the RDF histogram.

    """
    def __init__(self, check_thermo_every, check_rdf_every, rdf_dr):
        self.check_thermo_every = check_thermo_every
        self.check_rdf_every = check_rdf_every
        self.rdf_dr = rdf_dr

    def __call__(self, sim):
        with sim.hoomd:
            # thermodynamic properties
            if sim.dimension == 3:
                quantities_logged = ['temperature','pressure','lx','ly','lz','xy','xz','yz']
            elif sim.dimension == 2:
                quantities_logged = ['temperature','pressure','lx','ly','xy']
            else:
                raise ValueError('HOOMD only supports 2d or 3d simulations')
            sim[self].logger = hoomd.analyze.log(filename=None,
                                                 quantities= quantities_logged,
                                                 period=self.check_thermo_every)
            sim[self].thermo_callback = AddEnsembleAnalyzer.ThermodynamicsCallback(sim[self].logger)
            hoomd.analyze.callback(callback=sim[self].thermo_callback,
                                   period=self.check_thermo_every)

            # pair distribution function
            rdf_params = PairMatrix(sim.types)
            rmax = sim.potentials.pair.r[-1]
            bins = numpy.round(rmax/self.rdf_dr).astype(int)
            for pair in rdf_params:
                rdf_params[pair] = {'bins': bins, 'rmax': rmax}
            sim[self].rdf_callback = AddEnsembleAnalyzer.RDFCallback(sim[sim.initializer].system,rdf_params)
            hoomd.analyze.callback(callback=sim[self].rdf_callback,
                                   period=self.check_rdf_every)

    def extract_ensemble(self, sim):
        """Extract the average ensemble from a simulation instance.

        Parameters
        ----------
        sim : :class:`~relentless.simulate.simulate.SimulationInstance`
            The simulation.

        Returns
        -------
        :class:`~relentless.ensemble.Ensemble`
            Average ensemble from the simulation data.

        """
        rdf_recorder = sim[self].rdf_callback
        thermo_recorder = sim[self].thermo_callback
        ens = ensemble.Ensemble(T=thermo_recorder.T,
                                N=rdf_recorder.N,
                                V=thermo_recorder.V,
                                P=thermo_recorder.P)
        for pair in rdf_recorder.rdf:
            ens.rdf[pair] = rdf_recorder.rdf[pair]

        return ens

    class ThermodynamicsCallback:
        """HOOMD callback for averaging thermodynamic properties.

        Parameters
        ----------
        logger : :mod:`hoomd.analyze` logger
            Logger from which to retrieve data.

        """
        def __init__(self, logger):
            self.logger = logger
            self.reset()

        def __call__(self, timestep):
            self.num_samples += 1

            T = self.logger.query('temperature')
            self._T += T

            P = self.logger.query('pressure')
            self._P += P

            for key in self._V:
                val = self.logger.query(key.lower())
                self._V[key] += val

        def reset(self):
            """Resets sample number, ``T``, ``P``, and all ``V`` parameters to 0."""
            self.num_samples = 0
            self._T = 0.
            self._P = 0.
            if hasattr(self.logger, "Lz"):
                self._V = {'Lx' : 0., 'Ly': 0., 'Lz': 0., 'xy': 0., 'xz': 0., 'yz': 0.}
            else:
                self._V = {'Lx' : 0., 'Ly': 0., 'xy': 0.}

        @property
        def T(self):
            """float: Average temperature across samples."""
            if self.num_samples > 0:
                return self._T / self.num_samples
            else:
                return None

        @property
        def P(self):
            """float: Average pressure across samples."""
            if self.num_samples > 0:
                return self._P / self.num_samples
            else:
                return None

        @property
        def V(self):
            """float: Average extent across samples."""
            if self.num_samples > 0:
                _V = {key: self._V[key]/self.num_samples for key in self._V}
                if hasattr(self.logger, "Lz"):
                    return extent.TriclinicBox(**_V,convention=extent.TriclinicBox.Convention.HOOMD)                
                else:
                    return extent.ObliqueArea(**_V,convention=extent.ObliqueArea.Convention.HOOMD)
            else:
                return None

    class RDFCallback:
        """HOOMD callback for averaging radial distribution function across timesteps.

        Parameters
        ----------
        system : :mod:`hoomd.data` system
            Simulation system object.
        params : :class:`~relentless.collections.PairMatrix`
            Parameters to be used to initialize an instance of :class:`freud.density.RDF`.

        """
        def __init__(self, system, params):
            self.system = system

            self.num_samples = 0
            self._N = FixedKeyDict(params.types)
            for i in self._N:
                self._N[i] = 0

            self._rdf = PairMatrix(params.types)
            for i,j in self._rdf:
                self._rdf[i,j] = freud.density.RDF(bins=params[i,j]['bins'],
                                                   r_max=params[i,j]['rmax'],
                                                   normalize=(i==j))

        def __call__(self, timestep):
            snap = self.system.take_snapshot()
            if mpi.world.rank == 0:
                box_array = numpy.array([snap.box.Lx, snap.box.Ly, snap.box.Lz, snap.box.xy, snap.box.xz, snap.box.yz])
                if snap.box.dimensions == 2:
                    box_array[2] = 0.
                    box_array[-2:] = 0.
                box = freud.box.Box.from_box(box_array, dimensions=snap.box.dimensions)
                # pre build aabbs per type and count particles by type
                aabbs = {}
                type_masks = {}
                for i in self._N:
                    type_masks[i] = (snap.particles.typeid == snap.particles.types.index(i))
                    self._N[i] += numpy.sum(type_masks[i])
                    aabbs[i] = freud.locality.AABBQuery(box,snap.particles.position[type_masks[i]])
                # then do rdfs using the AABBs
                for i,j in self._rdf:                  
                    query_args = dict(self._rdf[i,j].default_query_args)
                    query_args.update(exclude_ii=(i==j))
                    # resetting when the samples are zero clears the RDF, on delay
                    self._rdf[i,j].compute(aabbs[j],
                                           snap.particles.position[type_masks[i]],
                                           neighbors=query_args,
                                           reset=(self.num_samples == 0))
            self.num_samples += 1

        @property
        def N(self):
            """:class:`~relentless.collections.FixedKeyDict`: Number of particles by type."""
            if self.num_samples > 0:
                N = FixedKeyDict(self._N.keys())               
                for i in self._N:
                    if mpi.world.rank == 0:
                        Ni = self._N[i]/self.num_samples
                    else:
                        Ni = None
                    Ni = mpi.world.bcast(Ni, root=0)
                    N[i] = Ni
                return N
            else:
                return None

        @property
        def rdf(self):
            """:class:`~relentless.collections.PairMatrix`: Radial distribution functions."""
            if self.num_samples > 0:
                rdf = PairMatrix(self._rdf.types)
                for pair in rdf:
                    if mpi.world.rank == 0:
                        gr = numpy.column_stack((self._rdf[pair].bin_centers,self._rdf[pair].rdf))
                    else:
                        gr = None
                    gr = mpi.world.bcast_numpy(gr,root=0)
                    rdf[pair] = ensemble.RDF(gr[:,0],gr[:,1])
                return rdf
            else:
                return None

        def reset(self):
            """Reset the averages."""
            self.num_samples = 0
            for i in self._N.types:
                self._N[i] = 0
            # rdf will be reset on next call

class HOOMD(simulate.Simulation):
    """Simulation using HOOMD-blue.

    A simulation is performed using `HOOMD-blue <https://hoomd-blue.readthedocs.io/en/stable>`_.
    HOOMD-blue is a molecular dynamics program that can execute on both CPUs and
    GPUs, as a single process or with MPI parallelism. The launch configuration
    will be automatically selected for you when the simulation is run.

    Currently, this class only supports operations for HOOMD 2.x. Support for
    HOOMD 3.x will be added in future. The `freud <https://freud.readthedocs.io>`_
    analysis package (version 2.x) is also required for initialization and analysis.
    To use this simulation backend, you will need to install both `hoomd` and
    `freud` into your Python environment. `hoomd` is available through
    `conda-forge` or can be built from source, while `freud` is available through
    both `pip` and `conda-forge`. Please refer to the package documentation for
    details of how to install these.

    Raises
    ------
    ImportError
        If the :mod:`hoomd` package is not found or is not version 2.x.
    ImportError
        If the :mod:`freud` package is not found or is not version 2.x.

    """
    def __init__(self, initializer, operations=None):
        if not _hoomd_found:
            raise ImportError('HOOMD not found.')
        elif version.parse(hoomd.__version__).major != 2:
            raise ImportError('Only HOOMD 2.x is supported.')

        if not _freud_found:
            raise ImportError('freud not found.')
        elif version.parse(freud.__version__).major != 2:
            raise ImportError('Only freud 2.x is supported.')

        super().__init__(initializer, operations)

    def _new_instance(self, initializer, potentials, directory):
        sim = super()._new_instance(initializer, potentials, directory)

        # initialize hoomd exec conf once
        if hoomd.context.exec_conf is None:
            hoomd.context.initialize('--notice-level=0')
            hoomd.util.quiet_status()

        # initialize
        sim.hoomd = hoomd.context.SimulationContext()
        initializer(sim)
        sim.dimension = sim[initializer].system.box.dimensions
        sim.types = sim[initializer].system.particles.types

        # attach the potentials
        self._attach_potentials(sim)

        return sim

    def _attach_potentials(self, sim):
        """Attach potentials to the simulation.

        The potentials are attached to the instance as the last step of
        initialization, after the particles types are encoded by the initializer.

        Parameters
        ----------
        sim : :class:`~relentless.simulate.SimulationInstance`
            Simulation instance.

        """
        def _table_eval(r_i, rmin, rmax, **coeff):
            r = coeff['r']
            u = coeff['u']
            f = coeff['f']
            u_r = Interpolator(r,u)
            f_r = Interpolator(r,f)
            return (u_r(r_i), f_r(r_i))
        with sim.hoomd:
            neighbor_list = hoomd.md.nlist.tree(r_buff=sim.potentials.pair.neighbor_buffer)
            pair_potential = hoomd.md.pair.table(width=len(sim.potentials.pair.r),
                                                 nlist=neighbor_list)
            for i,j in sim.pairs:
                r = sim.potentials.pair.r
                u = sim.potentials.pair.energy((i,j))
                f = sim.potentials.pair.force((i,j))
                pair_potential.pair_coeff.set(i,j,
                                              func=_table_eval,
                                              rmin=r[0],
                                              rmax=r[-1],
                                              coeff=dict(r=r,u=u,f=f))

    # initialization
    InitializeFromFile = InitializeFromFile
    InitializeRandomly = InitializeRandomly

    # energy minimization
    MinimizeEnergy = MinimizeEnergy

    # md integrators
    AddBrownianIntegrator = AddBrownianIntegrator
    RemoveBrownianIntegrator = RemoveBrownianIntegrator
    AddLangevinIntegrator = AddLangevinIntegrator
    RemoveLangevinIntegrator = RemoveLangevinIntegrator
    AddVerletIntegrator = AddVerletIntegrator
    RemoveVerletIntegrator = RemoveVerletIntegrator

    # run commands
    Run = Run
    RunUpTo = RunUpTo

    # analysis
    AddEnsembleAnalyzer = AddEnsembleAnalyzer
