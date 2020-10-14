import os
from packaging import version

import numpy as np

from relentless.core.collections import PairMatrix
from relentless.core.ensemble import RDF
from relentless.core.math import Interpolator
from relentless.core.volume import TriclinicBox
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
    """:py:class:`Simulation` using HOOMD framework.

    Raises
    ------
    ImportError
        If `hoomd` package is not found, or is not version 2.x.
    ImportError
        If `freud` package  is not found, or is not version 2.x.

    """
    def __init__(self, operations, **options):
        if not _hoomd_found:
            raise ImportError('HOOMD not found.')
        elif version.parse(hoomd.__version__).major != 2:
            raise ImportError('Only HOOMD 2.x is supported.')

        if not _freud_found:
            raise ImportError('freud not found.')
        elif version.parse(freud.__version__).major != 2:
            raise ImportError('Only freud 2.x is supported.')

        super().__init__(operations,**options)

        # initialize hoomd exec conf once
        if hoomd.context.exec_conf is None:
            hoomd.context.initialize('')

    def _new_instance(self, ensemble, potentials, directory):
        sim = super()._new_instance(ensemble,potentials,directory,**self.options)
        sim.context = hoomd.context.SimulationContext()
        sim.system = None
        return sim

## initializers
class Initialize(simulate.SimulationOperation):
    """:py:class:`SimulationOperation` that initializes a simulation box and pair potentials.

    Parameters
    ----------
    r_buff : float
        Buffer width (defaults to 0.4).

    """
    def __init__(self, r_buff=0.4):
        self.r_buff = r_buff

    def extract_box_params(self, sim):
        """Extracts HOOMD box parameters (*Lx*, *Ly*, *Lz*, *xy*, *xz*, *yz*)
        from the simulation's ensemble volume.

        Parameters
        ----------
        sim : :py:class:`Simulation`
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
            If the volume does not derive from :py:class:`TriclinicBox`.

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
        sim : :py:class:`Simulation`
            Simulation object.

        Returns
        -------
        `hoomd.data` snapshot
            Particle simulation snapshot.
        :py:class:`freud.Box`
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
        sim : :py:class:`Simulation`
            Simulation object.

        Raises
        ------
        ValueError
            If the length of the *r*, *u*, and *f* arrays are not all equal.

        """
        # first write all potentials to disk
        os.chdir(sim.directory.path)
        table_size = None
        files = PairMatrix(sim.ensemble.types)
        for i,j in sim.potentials:
            r = sim.potentials[i,j].get('r')
            u = sim.potentials[i,j].get('u')
            f = sim.potentials[i,j].get('f')
            if f is None:
                ur = Interpolator(r,u)
                f  = -ur.derivative(r,1)

            # validate table size
            if table_size is None:
                table_size = len(r)
            if len(r) != table_size or len(u) != table_size or len(f) != table_size:
                raise ValueError('HOOMD requires equal sized tables.')

            files[i,j] = sim.directory.file('table_{i}_{j}.dat'.format(i=i,j=j))
            header = '# Tabulated pair for ({i},{j})\n'.format(i=i,j=j)
            header += '# r u f'
            np.savetxt(files[i,j],
                       np.column_stack((r,u,f)),
                       header=header,
                       comments='')

        # create potentials in HOOMD script
        with sim.context:
            sim[self].neighbor_list = hoomd.md.nlist.tree(r_buff=self.r_buff)
            sim[self].pair_potential = hoomd.md.pair.table(width=table_size,
                                                           nlist=sim[self].neighbor_list)
            for i,j in files:
                with open(files[i,j]) as f:
                    r,u,f = self._extract_table(f.name)
                    sim[self].pair_potential.pair_coeff.set(i,j,func=self._table_eval,
                                                            rmin=r[0],rmax=r[-1],
                                                            coeff=dict(r=r,u=u,f=f))

    #helper methods for attach_potentials
    def _extract_table(self, filename):
        r = []
        u = []
        f = []
        with open(filename) as n:
            for line in n.readlines():
                line = line.strip()
                if line[0] == '#':
                    continue
                cols = line.split()
                values = [float(i) for i in cols]
                r.append(values[0])
                u.append(values[1])
                f.append(values[2])
        return (r, u, f)

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
        Options for file reading (as used in :py:func:`hoomd.init.read_gsd()`).

    """
    def __init__(self, filename, **options):
        self.filename = os.path.realpath(filename)
        self.options = options
        super().__init__()

    def __call__(self, sim):
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
        The seed to randomly initialize the particle locations (defaults to `None`).

    """
    def __init__(self, seed=None):
        self.seed = seed
        super().__init__()

    def __call__(self, sim):
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
                rs = np.random.uniform(size=(snap.particles.N,3))
                snap.particles.position[:] = box.make_absolute(rs)

                # set types of each
                snap.particles.typeid[:] = np.repeat(np.arange(len(sim.ensemble.types)),
                                                    [sim.ensemble.N[t] for t in sim.ensemble.types])

                # assume unit mass and thermalize to Maxwell-Boltzmann distribution
                snap.particles.mass[:] = 1.0
                vel = np.random.normal(scale=np.sqrt(sim.ensemble.kT),size=3)
                snap.particles.velocity[:] = vel-np.mean(vel)

                # read snapshot
                sim.system = hoomd.init.read_snapshot(snap)
        finally:
            # always restore old state if it exists
            if old_state is not None:
                np.random.set_state(old_state)

        self.attach_potentials(sim)

## integrators
class MinimizeEnergy(simulate.SimulationOperation):
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
        self.energy_tolerance = energy_tolerance
        self.force_tolerance = force_tolerance
        self.max_iterations = max_iterations
        self.dt = dt

    def __call__(self, sim):
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
    """:py:class:`SimulationOperation` to add an integrator (for equations of motion) to the simulation.

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
        sim : :py:class:`Simulation`
            Simulation object.

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
    """:py:class:`SimulationOperation` that removes a specified integration operation.

    Parameters
    ----------
    add_op : :py:class:`SimulationOperation`
        The addition/integration operation to be removed.

    """
    def __init__(self, add_op):
        self.add_op = add_op

    def __call__(self, sim):
        sim[self.add_op].integrator.disable()

class AddBrownianIntegrator(AddMDIntegrator):
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
        Options used in :py:func:`hoomd.md.integrate.brownian()`

    """
    def __init__(self, dt, friction, seed, **options):
        super().__init__(dt)
        self.friction = friction
        self.seed = seed
        self.options = options

    def __call__(self, sim):
        if not sim.ensemble.aka('NVT'):
            raise ValueError('Simulation ensemble is not NVT.')

        self.attach_integrator(sim)
        with sim.context:
            all_ = hoomd.group.all()
            sim[self].integrator = hoomd.md.integrate.brownian(group=all_,
                                                               kT=sim.ensemble.kT,
                                                               seed=seed,
                                                               **self.options)
            for t in sim.ensemble.N:
                try:
                    gamma = self.friction[t]
                except TypeError:
                    gamma = self.friction
                sim[self].integrator.set_gamma(t,gamma)

class RemoveBrownianIntegrator(RemoveMDIntegrator):
    """Removes the Brownian integrator operation.

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
        Time step size for each simulation iteration
    friction : float
        Sets drag coefficient for each particle type.
    seed : int
        Seed used to randomly generate a uniform force.
    options : kwargs
        Options used in :py:func:`hoomd.md.integrate.langevin()`

    """
    def __init__(self, dt, friction, seed, **options):
        super().__init__(dt)
        self.friction = friction
        self.seed = seed
        self.options = options

    def __call__(self, sim):
        if not sim.ensemble.aka('NVT'):
            raise ValueError('Simulation ensemble is not NVT.')

        self.attach_integrator(sim)
        with sim.context:
            all_ = hoomd.group.all()
            sim[self].integrator = hoomd.md.integrate.langevin(group=all_,
                                                               kT=sim.ensemble.kT,
                                                               seed=seed,
                                                               **self.options)
            for t in sim.ensemble:
                try:
                    gamma = self.friction[t]
                except TypeError:
                    gamma = self.friction
                sim[self].integrator.set_gamma(t,gamma)

class RemoveLangevinIntegrator(RemoveMDIntegrator):
    """Removes the Langevin integrator operation.

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
        Time step size for each simulation iteration
    tau_T : float
        Coupling constant for the thermostat.
    tau_P : float
        Coupling constant for the barostat.
    options : kwargs
        Options used in :py:func:`hoomd.md.integrate.npt()`

    """
    def __init__(self, dt, tau_T, tau_P, **options):
        super().__init__(dt)
        self.tau_T = tau_T
        self.tau_P = tau_P
        self.options = options

    def __call__(self, sim):
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
    dt : float
        Time step size for each simulation iteration
    tau_T : float
        Coupling constant for the thermostat.
    options : kwargs
        Options used in :py:func:`hoomd.md.integrate.npt()`

    """
    def __init__(self, dt, tau_T, **options):
        super().__init__(dt)
        self.tau_T = tau_T
        self.options = options

    def __call__(self, sim):
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
    """Removes the NPT integrator operation.

    Raises
    ------
    TypeError
        If the specified addition operation is not a NPT integrator.

    """
    def __init__(self, add_op):
        if not isinstance(add_op, AddNVTIntegrator):
            raise TypeError('Addition operation is not AddNVTIntegrator.')
        super().__init__(add_op)

class Run(simulate.SimulationOperation):
    """:py:class:`SimulationOperation` that advances the simulation by a given number of time steps.

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
    """:py:class:`SimulationOperation` that advances the simulation up to a given time step number.

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
        self.num_samples += 1
        self._T += self.logger.query('temperature')
        self._P += self.logger.query('pressure')
        for key in self._V:
            self._V[key] += self.logger.query(key.lower())

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
            vol = TriclinicBox(**_V,convention=TriclinicBox.Convention.HOOMD)
        else:
            return None

class RDFCallback:
    """HOOMD callback for averaging radial distribution function across timesteps.

    Parameters
    ----------
    system : `hoomd.data` snapshot
        Simulation system object.
    params : dict
        Parameters to be used to initialize an instance of :py:class:`freud.density.RDF`.

    """
    def __init__(self, system, params):
        self.system = system
        self.rdf = PairMatrix(params.types)
        for i,j in self.rdf:
            self.rdf[i,j] = freud.rdf.RDF(**params[i,j],normalize=(i==j))

    def __call__(self, timestep):
        with sim.context:
            hoomd.util.quiet_status()
            snap = self.system.take_snapshot()
            hoomd.util.unquiet_status()

            box = freud.box.Box.from_box(snap.configuration.box)
            for i,j in self._rdf:
                typei = (snap.particles.typeid == snap.particles.types.index(i))
                typej = (snap.particles.typeid == snap.particles.types.index(j))
                aabb = freud.locality.AABBQuery(box,snap.particles.position[typej])
                self.rdf[i,j].compute(aabb,
                                      snap.particles.position[typei],
                                      reset=False)

class AddEnsembleAnalyzer(simulate.SimulationOperation):
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
        self.check_thermo_every = check_thermo_every
        self.check_rdf_every = check_rdf_every
        self.rdf_dr = rdf_dr

    def __call__(self, sim):
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
            for pair in rdf_params:
                rmax = sim.potentials[pair]['r'][-1]
                bins = np.round(rmax/self.rdf_dr).astype(int)
                rdf_params[pair] = {'bins': bins, 'rmax': rmax}
            sim[self].rdf_callback = RDFCallback(sim.system,rdf_params)
            hoomd.analyze.callback(callback=sim[self].rdf_callback,
                                   period=self.check_rdf_every)

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
        ens = sim.ensemble.copy()
        ens.clear()

        thermo_recorder = sim[self].thermo_callback
        ens.T = thermo_recorder.T
        ens.P = thermo_recorder.P
        ens.V = thermo_recorder.V

        rdf_recorder = sim[self].rdf_callback
        for pair in rdf_recorder:
            ens.rdf[pair] = RDF(rdf_recorder[pair].bin_centers,
                                rdf_recorder[pair].rdf)

        return ens