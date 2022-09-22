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
import os
from packaging import version

import numpy

from relentless.collections import PairMatrix
from relentless.ensemble import RDF
from relentless.math import Interpolator
from relentless import mpi
from relentless.extent import Area, TriclinicBox, Volume
from relentless.extent import ObliqueArea
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
    """Initializes a simulation box and pair potentials."""
    def __call__(self, sim):
        # check the dimension of the simulation up front
        if isinstance(sim.ensemble.V, TriclinicBox):
            sim.dimension = 3
        elif isinstance(sim.ensemble.V, ObliqueArea):
            sim.dimension = 2
        else:
            raise TypeError('HOOMD boxes must be derived from TriclinicBox or ObliqueArea')

    def extract_box_params(self, sim):
        """Extracts HOOMD box parameters (``Lx``, ``Ly``, ``Lz``, ``xy``, ``xz``, ``yz``)
        from the simulation's ensemble extent.

        Parameters
        ----------
        sim : :class:`~relentless.simulate.simulate.Simulation`
            Simulation object.

        Returns
        -------
        array_like
            Array of the simulation box parameters.

        Raises
        ------
        ValueError
            If the extent is not set.
        TypeError
            If the extent does not derive from :class:`~relentless.extent.TriclinicBox` or :class:`~relentless.extent.ObliqueArea`.

        """
        # cast simulation box in HOOMD parameters
        V = sim.ensemble.V
        if V is None:
            raise ValueError('Box extent must be set.')
        elif not isinstance(V, (TriclinicBox, ObliqueArea)):
            raise TypeError('HOOMD boxes must be derived from TriclinicBox or ObliqueArea')
        if isinstance(V, TriclinicBox):
            Lx = V.a[0]
            Ly = V.b[1]
            Lz = V.c[2]
            xy = V.b[0]/Ly
            xz = V.c[0]/Lz
            yz = V.c[1]/Lz
        elif isinstance(V, ObliqueArea):
            Lx = V.a[0]
            Ly = V.b[1]
            Lz = 1
            xy = V.b[0]/Ly
            xz = 0
            yz = 0
        return Lx,Ly,Lz,xy,xz,yz

    def make_snapshot(self, sim):
        """Creates a particle snapshot and box for the simulation context.

        Parameters
        ----------
        sim : :class:`~relentless.simulate.simulate.Simulation`
            The simulation object.

        Returns
        -------
        :mod:`hoomd.data` snapshot
            Particle simulation snapshot.
        :class:`freud.box.Box`
            Particle simulation box.

        Raises
        ------
        ValueError
            If the particle number is not set.

        """
        # get total number of particles
        N = 0
        for t in sim.ensemble.types:
            if sim.ensemble.N[t] is None:
                raise ValueError('Number of particles for type {} must be set.'.format(t))
            N += sim.ensemble.N[t]

        # cast simulation box in HOOMD parameters
        Lx,Ly,Lz,xy,xz,yz = self.extract_box_params(sim)
        # make the empty snapshot in the current context
        with sim.context:
            box = hoomd.data.boxdim(Lx=Lx,Ly=Ly,Lz=Lz,xy=xy,xz=xz,yz=yz,dimensions=sim.dimension)
            snap = hoomd.data.make_snapshot(N=N,
                                            box=box,
                                            particle_types=list(sim.ensemble.types))
            # freud boxes are more useful than HOOMD boxes, so prefer that type
            box = freud.Box.from_box(box)

        return snap,box

    def attach_potentials(self, sim):
        """Adds tabulated pair potentials to simulation object.

        Parameters
        ----------
        sim : :class:`~relentless.simulate.simulate.Simulation`
            The simulation object.

        Raises
        ------
        ValueError
            If the length of the ``r``, ``u``, and ``f`` arrays are not all equal.

        """
        with sim.context:
            # create potentials in HOOMD script
            sim[self].neighbor_list = hoomd.md.nlist.tree(r_buff=sim.potentials.pair.neighbor_buffer)
            sim[self].pair_potential = hoomd.md.pair.table(width=len(sim.potentials.pair.r),
                                                           nlist=sim[self].neighbor_list)
            for i,j in sim.ensemble.pairs:
                r = sim.potentials.pair.r
                u = sim.potentials.pair.energy((i,j))
                f = sim.potentials.pair.force((i,j))
                sim[self].pair_potential.pair_coeff.set(i,j,
                                                        func=self._table_eval,
                                                        rmin=r[0],
                                                        rmax=r[-1],
                                                        coeff=dict(r=r,u=u,f=f))

    # helper method for attach_potentials
    def _table_eval(self, r_i, rmin, rmax, **coeff):
        r = coeff['r']
        u = coeff['u']
        f = coeff['f']
        u_r = Interpolator(r,u)
        f_r = Interpolator(r,f)
        return (u_r(r_i), f_r(r_i))

class InitializeFromFile(Initialize):
    """Initializes a simulation box and pair potentials from a GSD file.

    Parameters
    ----------
    filename : str
        The file from which to read the system data.
    options : kwargs
        Options for file reading (as used in :func:`hoomd.init.read_gsd`).

    Raises
    ------
    ValueError
        If the simulation box dimensions specified by the file is inconsistent
        with the ensemble volume (for a constant volume simulation).

    """
    def __init__(self, filename, **options):
        super().__init__()
        self.filename = os.path.realpath(filename)
        self.options = options

    def __call__(self, sim):
        super().__call__(sim)
        with sim.context:
            sim.system = hoomd.init.read_gsd(self.filename,**self.options)

            # check that the boxes are consistent in constant extent sims.
            if sim.ensemble.V is not None:
                system_box = sim.system.box
                box_from_file = numpy.array([system_box.Lx,
                                          system_box.Ly,
                                          system_box.Lz,
                                          system_box.xy,
                                          system_box.xy,
                                          system_box.yz])
                box_size = self.extract_box_params(sim)
                if not numpy.all(numpy.isclose(box_from_file,box_size)) or system_box.dimensions != sim.dimension:
                    raise ValueError('Box from file is is inconsistent with ensemble extent.')
        self.attach_potentials(sim)

class InitializeRandomly(Initialize):
    """Initializes a randomly generated simulation box and pair potentials.

    Places particles in random coordinates, sets particle types, gives the
    particles unit mass and thermalizes to the Maxwell-Boltzmann distribution.

    Parameters
    ----------
    seed : int
        The seed to randomly initialize the particle locations.

    """
    def __init__(self, seed):
        super().__init__()
        self.seed = seed

    def __call__(self, sim):
        super().__call__(sim)
        # if setting seed, preserve the current RNG state
        if self.seed is not None:
            old_state = numpy.random.get_state()
            numpy.random.seed(self.seed)
        else:
            old_state = None

        try:
            # randomly place the particles
            with sim.context:
                snap,box = self.make_snapshot(sim)
                # randomly place particles in fractional coordinates
                if mpi.world.rank == 0:
                    if sim.dimension == 3:
                        rs = numpy.random.uniform(size=(snap.particles.N,3))
                    elif sim.dimension == 2:
                        rs = numpy.zeros((snap.particles.N,3))
                        rs[:,:2] = numpy.random.uniform(size=(snap.particles.N,2))
                    else:
                        raise ValueError('HOOMD supports 2d and 3d simulations')
                    snap.particles.position[:] = box.make_absolute(rs)

                    # set types of each
                    snap.particles.typeid[:] = numpy.repeat(numpy.arange(len(sim.ensemble.types)),
                                                            [sim.ensemble.N[t] for t in sim.ensemble.types])

                    # assume unit mass and thermalize to Maxwell-Boltzmann distribution
                    snap.particles.mass[:] = 1.0
                    if sim.dimension == 3:
                        vel = numpy.random.normal(scale=numpy.sqrt(sim.ensemble.kT),size=(snap.particles.N,3))
                    elif sim.dimension == 2:
                        vel = numpy.zeros((snap.particles.N,3))
                        vel[:,:2] = numpy.random.normal(scale=numpy.sqrt(sim.ensemble.kT),size=(snap.particles.N,2))
                    else:
                        raise ValueError('HOOMD supports 2d and 3d simulations')
                    snap.particles.velocity[:] = vel-numpy.mean(vel,axis=0)

                # read snapshot
                sim.system = hoomd.init.read_snapshot(snap)
        finally:
            # always restore old state if it exists
            if old_state is not None:
                numpy.random.set_state(old_state)

        self.attach_potentials(sim)

## integrators
class MinimizeEnergy(simulate.SimulationOperation):
    """Runs an energy minimzation until converged.

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
        with sim.context:
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
        with sim.context:
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
    friction : float
        Sets drag coefficient for each particle type.
    seed : int
        Seed used to randomly generate a uniform force.
    options : kwargs
        Options used in :class:`hoomd.md.integrate.brownian`.

    """
    def __init__(self, dt, friction, seed, **options):
        super().__init__(dt)
        self.friction = friction
        self.seed = seed
        self.options = options

    def __call__(self, sim):
        self.attach_integrator(sim)
        with sim.context:
            all_ = hoomd.group.all()
            sim[self].integrator = hoomd.md.integrate.brownian(group=all_,
                                                               kT=sim.ensemble.kT,
                                                               seed=self.seed,
                                                               **self.options)
            for t in sim.ensemble.N:
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
    friction : float or dict
        Drag coefficient for each particle type (shared or per-type).
    seed : int
        Seed used to randomly generate a uniform force.
    options : kwargs
        Options used in :class:`hoomd.md.integrate.langevin`.

    """
    def __init__(self, dt, friction, seed, **options):
        super().__init__(dt)
        self.friction = friction
        self.seed = seed
        self.options = options

    def __call__(self, sim):
        self.attach_integrator(sim)
        with sim.context:
            all_ = hoomd.group.all()
            sim[self].integrator = hoomd.md.integrate.langevin(group=all_,
                                                               kT=sim.ensemble.kT,
                                                               seed=self.seed,
                                                               **self.options)
            for t in sim.ensemble.types:
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
        with sim.context:
            all_ = hoomd.group.all()
            if self.thermostat is None and self.barostat is None:
                sim[self].integrator = hoomd.md.integrate.nve(group=all_)
            elif isinstance(self.thermostat, simulate.BerendsenThermostat) and self.barostat is None:
                sim[self].integrator = hoomd.md.integrate.berendsen(group=all_,
                                                                    kT=sim.ensemble.kB*self.thermostat.T,
                                                                    tau=self.thermostat.tau)
            elif isinstance(self.thermostat, simulate.NoseHooverThermostat) and self.barostat is None:
                sim[self].integrator = hoomd.md.integrate.nvt(group=all_,
                                                              kT=sim.ensemble.kB*self.thermostat.T,
                                                              tau=self.thermostat.tau)
            elif self.thermostat is None and isinstance(self.barostat, simulate.MTKBarostat):
                sim[self].integrator = hoomd.md.integrate.nph(group=all_,
                                                              P=self.barostat.P,
                                                              tauP=self.barostat.tau)
            elif isinstance(self.thermostat, simulate.NoseHooverThermostat) and isinstance(self.barostat, simulate.MTKBarostat):
                sim[self].integrator = hoomd.md.integrate.npt(group=all_,
                                                              kT=sim.ensemble.kB*self.thermostat.T,
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
        with sim.context:
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
        with sim.context:
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
        with sim.context:
            # thermodynamic properties
            if isinstance(sim.ensemble.V, Volume):
                quantities_logged = ['temperature','pressure','lx','ly','lz','xy','xz','yz']
            if isinstance(sim.ensemble.V, Area):
                quantities_logged = ['temperature','pressure','lx','ly','xy']
            sim[self].logger = hoomd.analyze.log(filename=None,
                                                 quantities= quantities_logged,
                                                 period=self.check_thermo_every)
            sim[self].thermo_callback = AddEnsembleAnalyzer.ThermodynamicsCallback(sim[self].logger)
            hoomd.analyze.callback(callback=sim[self].thermo_callback,
                                   period=self.check_thermo_every)

            # pair distribution function
            rdf_params = PairMatrix(sim.ensemble.types)
            rmax = sim.potentials.pair.r[-1]
            bins = numpy.round(rmax/self.rdf_dr).astype(int)
            for pair in rdf_params:
                rdf_params[pair] = {'bins': bins, 'rmax': rmax}
            sim[self].rdf_callback = AddEnsembleAnalyzer.RDFCallback(sim.system,rdf_params)
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
        ens = sim.ensemble.copy()

        thermo_recorder = sim[self].thermo_callback
        ens.T = thermo_recorder.T
        ens.P = thermo_recorder.P
        ens.V = thermo_recorder.V

        rdf = sim[self].rdf_callback.rdf
        for pair in rdf:
            ens.rdf[pair] = rdf[pair]

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
            if not hasattr(self.logger, "Lz"):
                self._V = {'Lx' : 0., 'Ly': 0., 'xy': 0.}
            elif hasattr(self.logger, "Lz"):
                self._V = {'Lx' : 0., 'Ly': 0., 'Lz': 0., 'xy': 0., 'xz': 0., 'yz': 0.}

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
                if not hasattr(self.logger, "Lz"):
                    return ObliqueArea(**_V,convention=ObliqueArea.Convention.HOOMD)
                if hasattr(self.logger, "Lz"):
                    return TriclinicBox(**_V,convention=TriclinicBox.Convention.HOOMD)
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
            self._rdf = PairMatrix(params.types)
            for i,j in self._rdf:
                self._rdf[i,j] = freud.density.RDF(bins=params[i,j]['bins'],
                                                   r_max=params[i,j]['rmax'],
                                                   normalize=(i==j))

        def __call__(self, timestep):
            snap = self.system.take_snapshot()
            if mpi.world.rank == 0:
                box = freud.box.Box.from_box(snap.box)
                for i,j in self._rdf:
                    typei = (snap.particles.typeid == snap.particles.types.index(i))
                    typej = (snap.particles.typeid == snap.particles.types.index(j))
                    aabb = freud.locality.AABBQuery(box,snap.particles.position[typej])
                    query_args = dict(self._rdf[i,j].default_query_args)
                    query_args.update(exclude_ii=(i==j))
                    self._rdf[i,j].compute(aabb,
                                           snap.particles.position[typei],
                                           neighbors=query_args,
                                           reset=False)

        @property
        def rdf(self):
            rdf = PairMatrix(self._rdf.types)
            for pair in rdf:
                if mpi.world.rank == 0:
                    gr = numpy.column_stack((self._rdf[pair].bin_centers,self._rdf[pair].rdf))
                else:
                    gr = None
                gr = mpi.world.bcast_numpy(gr,root=0)
                rdf[pair] = RDF(gr[:,0],gr[:,1])
            return rdf

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
    def __init__(self, operations=None, **options):
        if not _hoomd_found:
            raise ImportError('HOOMD not found.')
        elif version.parse(hoomd.__version__).major != 2:
            raise ImportError('Only HOOMD 2.x is supported.')

        if not _freud_found:
            raise ImportError('freud not found.')
        elif version.parse(freud.__version__).major != 2:
            raise ImportError('Only freud 2.x is supported.')

        super().__init__(operations,**options)

    def _new_instance(self, ensemble, potentials, directory):
        sim = super()._new_instance(ensemble,potentials,directory)

        # initialize hoomd exec conf once
        if hoomd.context.exec_conf is None:
            hoomd.context.initialize('--notice-level=0')
            hoomd.util.quiet_status()

        sim.context = hoomd.context.SimulationContext()
        sim.system = None
        # default dimensionality is unknown in HOOMD
        sim.dimension = None
        return sim

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
