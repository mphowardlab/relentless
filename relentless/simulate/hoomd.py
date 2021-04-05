import os
from packaging import version

import numpy as np

from relentless._collections import PairMatrix
from relentless.ensemble import RDF
from relentless._math import Interpolator
from relentless.volume import TriclinicBox
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

class HOOMD(simulate.Simulation):
    """:class:`Simulation` using HOOMD framework.

    Raises
    ------
    ImportError
        If the `hoomd` package is not found, or is not version 2.x.
    ImportError
        If the `freud` package is not found, or is not version 2.x.

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

    def _new_instance(self, ensemble, potentials, directory, communicator):
        sim = super()._new_instance(ensemble,potentials,directory,communicator)

        # initialize hoomd exec conf once
        if hoomd.context.exec_conf is None:
            hoomd.context.initialize('--notice-level=0', mpi_comm=sim.communicator.comm)
            hoomd.util.quiet_status()
            HOOMD._communicator = sim.communicator
        elif sim.communicator is not HOOMD._communicator:
            # HOOMD 2.x does not allow changing the communicator
            raise ValueError('HOOMD-blue does not support changing communicators after first initialization')

        sim.context = hoomd.context.SimulationContext()
        sim.system = None
        return sim
HOOMD._communicator = None

## initializers
class Initialize(simulate.SimulationOperation):
    """Initializes a simulation box and pair potentials.

    Parameters
    ----------
    neighbor_buffer : float
        Buffer width.

    """
    def __init__(self, neighbor_buffer):
        self.neighbor_buffer = neighbor_buffer

    def extract_box_params(self, sim):
        """Extracts HOOMD box parameters (*Lx*, *Ly*, *Lz*, *xy*, *xz*, *yz*)
        from the simulation's ensemble volume.

        Parameters
        ----------
        sim : :class:`Simulation`
            Simulation object.

        Returns
        -------
        array_like
            Array of the simulation box parameters.

        Raises
        ------
        ValueError
            If the volume is not set.
        TypeError
            If the volume does not derive from :class:`TriclinicBox`.

        """
        # cast simulation box in HOOMD parameters
        V = sim.ensemble.V
        if V is None:
            raise ValueError('Box volume must be set.')
        elif not isinstance(V, TriclinicBox):
            raise TypeError('HOOMD boxes must be derived from TriclinicBox')

        Lx = V.a[0]
        Ly = V.b[1]
        Lz = V.c[2]
        xy = V.b[0]/Ly
        xz = V.c[0]/Lz
        yz = V.c[1]/Lz

        return np.array([Lx,Ly,Lz,xy,xz,yz])

    def make_snapshot(self, sim):
        """Creates a particle snapshot and box for the simulation context.

        Parameters
        ----------
        sim : :class:`Simulation`
            The simulation object.

        Returns
        -------
        `hoomd.data` snapshot
            Particle simulation snapshot.
        :class:`freud.Box`
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
            box = hoomd.data.boxdim(Lx=Lx,Ly=Ly,Lz=Lz,xy=xy,xz=xz,yz=yz)
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
        sim : :class:`Simulation`
            The simulation object.

        Raises
        ------
        ValueError
            If the length of the *r*, *u*, and *f* arrays are not all equal.

        """
        with sim.context:
            # create potentials in HOOMD script
            sim[self].neighbor_list = hoomd.md.nlist.tree(r_buff=self.neighbor_buffer)
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

    #helper method for attach_potentials
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
    neighbor_buffer : float
        Buffer width.
    options : kwargs
        Options for file reading (as used in :meth:`hoomd.init.read_gsd()`).

    """
    def __init__(self, filename, neighbor_buffer, **options):
        super().__init__(neighbor_buffer)
        self.filename = os.path.realpath(filename)
        self.options = options

    def __call__(self, sim):
        """Performs the from-file initialization operation.

        Parameters
        ----------
        sim : :class:`Simulation`
            The simulation object.

        Raises
        ------
        ValueError
            If the simulation box dimensions specified by the file is inconsistent
            with the ensemble volume (for a constant volume simulation).

        """
        with sim.context:
            sim.system = hoomd.init.read_gsd(self.filename,**self.options)

            # check that the boxes are consistent in constant volume sims.
            if sim.ensemble.constant['V']:
                system_box = sim.system.box
                box_from_file = np.array([system_box.Lx,
                                          system_box.Ly,
                                          system_box.Lz,
                                          system_box.xy,
                                          system_box.xy,
                                          system_box.yz])
                box_from_ensemble = self.extract_box_params(sim)
                if not np.all(np.isclose(box_from_file,box_from_ensemble)):
                    raise ValueError('Box from file is is inconsistent with ensemble volume.')

        self.attach_potentials(sim)

class InitializeRandomly(Initialize):
    """Initializes a randomly generated simulation box and pair potentials.

    Parameters
    ----------
    seed : int
        The seed to randomly initialize the particle locations.
    neighbor_buffer : float
        Buffer width.

    """
    def __init__(self, seed, neighbor_buffer):
        super().__init__(neighbor_buffer)
        self.seed = seed

    def __call__(self, sim):
        """Performs the random initialization operation.

        Places particles in random coordinates, sets particle types, gives the
        particles unit mass and thermalizes to the Maxwell-Boltzmann distribution.

        Parameters
        ----------
        sim : :class:`Simulation`
            The simulation object.

        """
        # if setting seed, preserve the current RNG state
        if self.seed is not None:
            old_state = np.random.get_state()
            np.random.seed(self.seed)
        else:
            old_state = None

        try:
            # randomly place the particles
            with sim.context:
                snap,box = self.make_snapshot(sim)

                # randomly place particles in fractional coordinates
                if sim.communicator.rank == 0:
                    rs = np.random.uniform(size=(snap.particles.N,3))
                    snap.particles.position[:] = box.make_absolute(rs)

                    # set types of each
                    snap.particles.typeid[:] = np.repeat(np.arange(len(sim.ensemble.types)),
                                                        [sim.ensemble.N[t] for t in sim.ensemble.types])

                    # assume unit mass and thermalize to Maxwell-Boltzmann distribution
                    snap.particles.mass[:] = 1.0
                    vel = np.random.normal(scale=np.sqrt(sim.ensemble.kT),size=(snap.particles.N,3))
                    snap.particles.velocity[:] = vel-np.mean(vel,axis=0)

                # read snapshot
                sim.system = hoomd.init.read_snapshot(snap)
        finally:
            # always restore old state if it exists
            if old_state is not None:
                np.random.set_state(old_state)

        self.attach_potentials(sim)

## integrators
class MinimizeEnergy(simulate.SimulationOperation):
    """Runs an energy minimzation until converged.

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
        self.energy_tolerance = energy_tolerance
        self.force_tolerance = force_tolerance
        self.max_iterations = max_iterations
        self.dt = dt

    def __call__(self, sim):
        """Performs the energy minimization operation.

        Parameters
        ----------
        sim : :class:`Simulation`
            The simulation object.

        Raises
        ------
        RuntimeError
            If energy minimization has failed to converge within the maximum
            number of iterations.

        """
        with sim.context:
            # setup FIRE minimization
            fire = hoomd.md.integrate.mode_minimize_fire(dt=self.dt,
                                                         Etol=self.energy_tolerance,
                                                         ftol=self.force_tolerance)
            all_ = hoomd.group.all()
            nve = hoomd.md.integrate.nve(all_)

            # run while not yet converged
            it = 0
            while not fire.has_converged() and it < self.max_iterations:
                hoomd.run(100)
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
        """Enables standard integration methods in the simulation context.

        Parameters
        ----------
        sim : :class:`Simulation`
            The simulation object.

        """
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
    add_op : :class:`SimulationOperation`
        The addition/integration operation to be removed.

    """
    def __init__(self, add_op):
        self.add_op = add_op

    def __call__(self, sim):
        """Removes the integrator from the simulation.

        Parameters
        ----------
        sim : :class:`Simulation`
            The simulation object.

        Raises
        ------
        AttributeError
            If the specified integrator has already been removed.

        """
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
        Options used in :class:`hoomd.md.integrate.brownian()`.

    """
    def __init__(self, dt, friction, seed, **options):
        super().__init__(dt)
        self.friction = friction
        self.seed = seed
        self.options = options

    def __call__(self, sim):
        """Adds the Brownian integrator to the simulation.

        Parameters
        ----------
        sim : :class:`Simulation`
            The simulation object.

        Raises
        ------
        ValueError
            If the simulation ensemble is not NVT.

        """
        if not sim.ensemble.aka('NVT'):
            raise ValueError('Simulation ensemble is not NVT.')

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
        if not isinstance(add_op, AddBrownianIntegrator):
            raise TypeError('Addition operation is not AddBrownianIntegrator.')
        super().__init__(add_op)

class AddLangevinIntegrator(AddMDIntegrator):
    """Langevin dynamics for a NVT ensemble.

    Parameters
    ----------
    dt : float
        Time step size for each simulation iteration.
    friction : float or dict
        Drag coefficient for each particle type (shared or per-type).
    seed : int
        Seed used to randomly generate a uniform force.
    options : kwargs
        Options used in :class:`hoomd.md.integrate.langevin()`.

    """
    def __init__(self, dt, friction, seed, **options):
        super().__init__(dt)
        self.friction = friction
        self.seed = seed
        self.options = options

    def __call__(self, sim):
        """Adds the Langevin integrator to the simulation.

        Parameters
        ----------
        sim : :class:`Simulation`
            The simulation object.

        Raises
        ------
        ValueError
            If the simulation ensemble is not NVT.

        """
        if not sim.ensemble.aka('NVT'):
            raise ValueError('Simulation ensemble is not NVT.')

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
    """Removes the Langevin integrator operation.

    Parameters
    ----------
    add_op : :class:`AddLangevinIntegrator`
        The integrator addition operation to be removed.

    Raises
    ------
    TypeError
        If the specified addition operation is not a Langevin integrator.

    """
    def __init__(self, add_op):
        if not isinstance(add_op, AddLangevinIntegrator):
            raise TypeError('Addition operation is not AddLangevinIntegrator.')
        super().__init__(add_op)

class AddNPTIntegrator(AddMDIntegrator):
    """NPT integration via MTK barostat-thermostat.

    Parameters
    ----------
    dt : float
        Time step size for each simulation iteration.
    tau_T : float
        Coupling constant for the thermostat.
    tau_P : float
        Coupling constant for the barostat.
    options : kwargs
        Options used in :class:`hoomd.md.integrate.npt()`.

    """
    def __init__(self, dt, tau_T, tau_P, **options):
        super().__init__(dt)
        self.tau_T = tau_T
        self.tau_P = tau_P
        self.options = options

    def __call__(self, sim):
        """Adds the NPT integrator to the simulation.

        Parameters
        ----------
        sim : :class:`Simulation`
            The simulation object.

        Raises
        ------
        ValueError
            If the simulation ensemble is not NPT.

        """
        if not sim.ensemble.aka('NPT'):
            raise ValueError('Simulation ensemble is not NPT.')

        self.attach_integrator(sim)
        with sim.context:
            all_ = hoomd.group.all()
            sim[self].integrator = hoomd.md.integrate.npt(group=all_,
                                                          kT=sim.ensemble.kT,
                                                          tau=self.tau_T,
                                                          P=sim.ensemble.P,
                                                          tauP=self.tau_P,
                                                          **self.options)

class RemoveNPTIntegrator(RemoveMDIntegrator):
    """Removes the NPT integrator operation.

    Parameters
    ----------
    add_op : :class:`AddNPTIntegrator`
        The integrator addition operation to be removed.

    Raises
    ------
    TypeError
        If the specified addition operation is not a NPT integrator.

    """
    def __init__(self, add_op):
        if not isinstance(add_op, AddNPTIntegrator):
            raise TypeError('Addition operation is not AddNPTIntegrator.')
        super().__init__(add_op)

class AddNVTIntegrator(AddMDIntegrator):
    r"""NVT integration via Nos\'e-Hoover thermostat.

    Parameters
    ----------
    add_op : :class:`AddNVTIntegrator`
        The integrator addition operation to be removed.

    Parameters
    ----------
    dt : float
        Time step size for each simulation iteration.
    tau_T : float
        Coupling constant for the thermostat.
    options : kwargs
        Options used in :class:`hoomd.md.integrate.nvt()`.

    """
    def __init__(self, dt, tau_T, **options):
        super().__init__(dt)
        self.tau_T = tau_T
        self.options = options

    def __call__(self, sim):
        """Adds the NVT integrator to the simulation.

        Parameters
        ----------
        sim : :class:`Simulation`
            The simulation object.

        Raises
        ------
        ValueError
            If the simulation ensemble is not NVT.

        """
        if not sim.ensemble.aka('NVT'):
            raise ValueError('Simulation ensemble is not NVT.')

        self.attach_integrator(sim)
        with sim.context:
            all_ = hoomd.group.all()
            sim[self].integrator = hoomd.md.integrate.nvt(group=all_,
                                                          kT=sim.ensemble.kT,
                                                          tau=self.tau_T,
                                                          **self.options)

class RemoveNVTIntegrator(RemoveMDIntegrator):
    """Removes the NVT integrator operation.

    Parameters
    ----------
    add_op : :class:`AddNVTIntegrator`
        The integrator addition operation to be removed.

    Raises
    ------
    TypeError
        If the specified addition operation is not a NVT integrator.

    """
    def __init__(self, add_op):
        if not isinstance(add_op, AddNVTIntegrator):
            raise TypeError('Addition operation is not AddNVTIntegrator.')
        super().__init__(add_op)

class Run(simulate.SimulationOperation):
    """Advances the simulation by a given number of time steps.

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
    """Advances the simulation up to a given time step number.

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
class ThermodynamicsCallback:
    """HOOMD callback for averaging thermodynamic properties.

    Parameters
    ----------
    logger : `hoomd.analyze` logger
        Logger from which to retrieve data.

    """
    def __init__(self, logger):
        self.logger = logger
        self.reset()

    def __call__(self, timestep):
        """Evaluates the callback.

        Parameters
        ----------
        timestep : int
            The timestep at which to evaluate the callback.

        """
        self.num_samples += 1

        T = self.logger.query('temperature')
        sim.communicator.bcast(T,root=0)
        self._T += T

        P = self.logger.query('P')
        sim.communicator.bcast(P,root=0)
        self._P += P

        for key in self._V:
            val = self.logger.query(key.lower())
            sim.communicator.bcast(val,root=0)
            self._V[key] += val

    def reset(self):
        """Resets sample number, *T*, *P*, and all *V* parameters to 0."""
        self.num_samples = 0
        self._T = 0.
        self._P = 0.
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
        """float: Average volume across samples."""
        if self.num_samples > 0:
            _V = {key: self._V[key]/self.num_samples for key in self._V}
            return TriclinicBox(**_V,convention=TriclinicBox.Convention.HOOMD)
        else:
            return None

class RDFCallback:
    """HOOMD callback for averaging radial distribution function across timesteps.

    Parameters
    ----------
    system : `hoomd.data` system
        Simulation system object.
    params : :class:`PairMatrix`
        Parameters to be used to initialize an instance of :class:`freud.density.RDF`.

    """
    def __init__(self, system, params, communicator):
        self.system = system
        self._rdf = PairMatrix(params.types)
        self.communicator = communicator
        for i,j in self.rdf:
            self._rdf[i,j] = freud.density.RDF(bins=params[i,j]['bins'],
                                               r_max=params[i,j]['rmax'],
                                               normalize=(i==j))

    def __call__(self, timestep):
        """Evaluates the callback.

        Parameters
        ----------
        timestep : int
            The timestep at which to evaluate the callback.

        """
        snap = self.system.take_snapshot()
        if self.communicator.rank == 0:
            box = freud.box.Box.from_box(snap.box)
            for i,j in self.rdf:
                typei = (snap.particles.typeid == snap.particles.types.index(i))
                typej = (snap.particles.typeid == snap.particles.types.index(j))
                aabb = freud.locality.AABBQuery(box,snap.particles.position[typej])
                self._rdf[i,j].compute(aabb,
                                       snap.particles.position[typei],
                                       reset=False)

    @property
    def rdf(self):
        rdf = PairMatrix(self._rdf.types)
        for pair in rdf:
            if self.communicator.rank == 0:
                gr = np.column_stack((self._rdf[pair].bin_centers,self._rdf[pair].rdf))
            else:
                gr = None
            self.communicator.bcast(gr,root=0)
            rdf[pair] = RDF(gr[:,0],gr[:,1])
        return rdf

class AddEnsembleAnalyzer(simulate.SimulationOperation):
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
        self.check_thermo_every = check_thermo_every
        self.check_rdf_every = check_rdf_every
        self.rdf_dr = rdf_dr

    def __call__(self, sim):
        """Adds the ensemble analyzer to the simulation.

        Parameters
        ----------
        sim : :class:`Simulation`
            The simulation object.

        Raises
        ------
        ValueError
            If the simulation ensemble does not have constant N.

        """
        if not all([sim.ensemble.constant['N'][t] for t in sim.ensemble.types]):
            return ValueError('This analyzer requires constant N.')

        with sim.context:
            # thermodynamic properties
            sim[self].logger = hoomd.analyze.log(filename=None,
                                                 quantities=['temperature',
                                                             'pressure',
                                                             'lx','ly','lz','xy','xz','yz'],
                                                 period=self.check_thermo_every)
            sim[self].thermo_callback = ThermodynamicsCallback(sim[self].logger)
            hoomd.analyze.callback(callback=sim[self].thermo_callback,
                                   period=self.check_thermo_every)

            # pair distribution function
            rdf_params = PairMatrix(sim.ensemble.types)
            rmax = sim.potentials.pair.r[-1]
            bins = np.round(rmax/self.rdf_dr).astype(int)
            for pair in rdf_params:
                rdf_params[pair] = {'bins': bins, 'rmax': rmax}
            sim[self].rdf_callback = RDFCallback(sim.system,rdf_params)
            hoomd.analyze.callback(callback=sim[self].rdf_callback,
                                   period=self.check_rdf_every)

    def extract_ensemble(self, sim):
        """Creates an ensemble with the averaged thermodynamic properties and rdf.

        Parameters
        ----------
        sim : :class:`Simulation`
            The simulation object.

        Returns
        -------
        :class:`Ensemble`
            Ensemble with averaged thermodynamic properties and rdf.

        """
        ens = sim.ensemble.copy()
        ens.clear()

        thermo_recorder = sim[self].thermo_callback
        ens.T = thermo_recorder.T
        ens.P = thermo_recorder.P
        ens.V = thermo_recorder.V

        rdf = sim[self].rdf_callback.rdf
        for pair in rdf:
            ens.rdf[pair] = rdf[pair]

        return ens
