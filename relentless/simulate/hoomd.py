import os
from packaging import version

import numpy as np

from relentless.core.collections import PairMatrix
from relentless.core.ensemble import RDF
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

    def _new_instance(self, ensemble, potentials, directory):
        sim = super()._new_instance(ensemble,potentials,directory,**self.options)
        sim.context = hoomd.SimulationContext()
        sim.system = None
        return sim

## initializers
class Initialize(simulate.SimulationOperation):
    def __init__(self, r_buff=0.4):
        self.r_buff = r_buff

    def extract_box_params(self, sim):
        # cast simulation box in HOOMD parameters
        V = sim.ensemble.volume
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
                                            particle_types=sim.ensemble.types)
            # freud boxes are more useful than HOOMD boxes, so prefer that type
            box = freud.Box.from_box(box)

        return snap,box

    def attach_potentials(self, sim):
        # first write all potentials to disk
        table_size = None
        files = PairMatrix(sim.ensemble.types)
        for i,j in potentials:
            r = sim.potentials[i,j]['r']
            u = sim.potentials[i,j]['u']
            f = sim.potentials[i,j]['f']

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
                sim[self].pair_potential.set_from_file(i,j,files[i,j])

class InitializeFromFile(Initialize):
    def __init__(self, filename, **options):
        self.filename = os.path.realpath(filename)
        self.options = options

    def __call__(self, sim):
        with sim.context:
            sim.system = hoomd.init.read_gsd(self.filename,**self.options)

            # check that the boxes are consistent in constant volume sims.
            if sim.ensemble.constant('V'):
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
    def __init__(self, seed=None):
        self.seed = seed

    def __call__(self, sim):
        # if setting seed, preserve the current RNG state
        if seed is not None:
            old_state = np.random.get_state()
            np.random.seed(seed)
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
                vel = np.random.normal(scale=np.sqrt(sim.ensemble.kT))
                snap.particles.velocity[:] = vel-np.mean(vel,axis=1)

                # read snapshot
                sim.system = hoomd.init.read_snapshot(snap)
        finally:
            # always restore old state if it exists
            if old_state is not None:
                np.random.set_state(old_state)

        self.attach_potentials(sim)

## integrators
class MinimizeEnergy(simulate.SimulationOperation):
    def __init__(self, energy_tolerance, force_tolerance, max_iterations, dt):
        self.energy_tolerance = energy_tolerance
        self.force_tolerance = force_tolerance
        self.max_iterations = max_iterations
        self.dt = dt

    def __call__(self, sim):
        with sim.context:
            # setup FIRE minimization
            fire = hoomd.md.integrate.mode_minimize_fire(dt=dt,
                                                         etol=self.energy_tolerance,
                                                         ftol=self.force_tolerance)
            all_ = hoomd.group.all()
            nve = hoomd.md.integrate.nve(all_)

            # run while not yet converged
            it = 0
            while not fire.has_converged() and iteration < self.max_iterations:
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
    def __init__(self, add_op):
        self.add_op = add_op

    def __call__(self, sim):
        sim[self.add_op].integrator.disable()

class AddBrownianIntegrator(AddMDIntegrator):
    """Brownian dynamics."""
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
            for t in sim.ensemble:
                try:
                    gamma = self.friction[t]
                except TypeError:
                    gamma = self.friction
                sim[self].integrator.set_gamma(t,gamma)

class RemoveBrownianIntegrator(RemoveMDIntegrator):
    def __init__(self, add_op):
        if not isinstance(add_op, AddBrownianIntegrator):
            raise TypeError('Addition operation is not AddBrownianIntegrator.')
        super().__init__(add_op)

class AddLangevinIntegrator(AddMDIntegrator):
    """Langevin dynamics."""
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
    def __init__(self, add_op):
        if not isinstance(add_op, AddLangevinIntegrator):
            raise TypeError('Addition operation is not AddLangevinIntegrator.')
        super().__init__(add_op)

class AddNPTIntegrator(AddMDIntegrator):
    """NPT velocity Verlet."""
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
    def __init__(self, add_op):
        if not isinstance(add_op, AddNPTIntegrator):
            raise TypeError('Addition operation is not AddNPTIntegrator.')
        super().__init__(add_op)

class AddNVTIntegrator(AddMDIntegrator):
    """NVT velocity Verlet."""
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
    def __init__(self, add_op):
        if not isinstance(add_op, AddNVTIntegrator):
            raise TypeError('Addition operation is not AddNVTIntegrator.')
        super().__init__(add_op)

class Run(simulate.SimulationOperation):
    """Advance the simulation."""
    def __init__(self, steps):
        self.steps = steps

    def __call__(self, sim):
        with sim.context:
            hoomd.run(self.steps)

class RunUpTo(simulate.SimulationOperation):
    def __init__(self, step):
        self.step = step

    def __call__(self, sim):
        with sim.context:
            hoomd.run_upto(self.step)

## analyzers
class ThermodynamicsCallback:
    """HOOMD callback for averaging thermodynamic properties."""
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
        self.num_samples = 0
        self._T = 0.
        self._P = 0.
        self._V = {'Lx' : 0., 'Ly': 0., 'Lz': 0., 'xy': 0., 'xz': 0., 'yz': 0.}

    @property
    def T(self):
        if self.num_samples > 0:
            return self._T / self.num_samples
        else:
            return None

    @property
    def P(self):
        if self.num_samples > 0:
            return self._P / self.num_samples
        else:
            return None

    @property
    def V(self):
        if self.num_samples > 0:
            _V = {key: self._V[key]/self.num_samples for key in self._V}
            vol = TriclinicBox(**_V,convention=TriclinicBox.Convention.HOOMD)
        else:
            return None

class RDFCallback:
    """HOOMD callback for averaging radial distribution function."""
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
                bins = np.round(rmax/dr).astype(int)
                rdf_params[pair] = {'bins': bins, 'rmax': rmax}
            sim[self].rdf_callback = RDFCallback(sim.system,rdf_params)
            hoomd.analyze.callback(callback=sim[self].rdf_callback,
                                   period=self.check_rdf_every)

    def extract_ensemble(self, sim):
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
