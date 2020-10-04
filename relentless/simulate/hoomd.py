from packaging import version

import numpy as np

from relentless.core.collections import PairMatrix
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

    def run(self,ensemble, potentials, directory):
        if not isinstance(self.operations[0],Initialize):
            raise TypeError('First operation must be a HOOMD initializer.')
        super().run(ensemble,potentials,directory)

## initializers

class Initialize(simulate.SimulationOperation):
    def __init__(self, r_buff=0.4):
        self.r_buff = r_buff

    def __call__(self, sim):
        # inject a context and system into the SimulationInstance
        sim.context = hoomd.SimulationContext()
        sim.system = None

    def make_snapshot(self, sim):
        # get total number of particles
        N = 0
        for t in sim.ensemble.types:
            if sim.ensemble.N[t] is None:
                raise ValueError('Number of particles for type {} must be set.'.format(t))
            N += sim.ensemble.N[t]

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
            nl = hoomd.md.nlist.tree(r_buff=self.r_buff)
            pot = hoomd.md.pair.table(width=table_size,
                                      nlist=nl)
            for i,j in files:
                pot.set_from_file(i,j,files[i,j])

class InitializeFromGSD(Initialize):
    def __init__(self, filename, **options):
        self.filename = filename
        self.options = options

    def __call__(self, sim):
        super().__call__(sim)

        with sim.context:
            sim.system = hoomd.init.read_gsd(filename,**self.options)
        self.attach_potentials(sim)

class InitializeRandomly(Initialize):
    def __init__(self, seed=None):
        self.seed = seed

    def __call__(self, sim):
        super().__call__(sim)

        with sim.context:
            snap,box = self.make_snapshot(sim)

            # randomly place particles in fractional coordinates
            if seed is not None:
                np.random.seed(seed)
            rs = np.random.uniform(size=(snap.particles.N,3))
            snap.particles.position[:] = box.make_absolute(rs)

            # set types of each
            snap.particles.types = tuple(sim.ensemble.types)
            snap.particles.typeid[:] = np.repeat(np.arange(len(sim.ensemble.types)),
                                                [sim.ensemble.N[t] for t in sim.ensemble.types])

            # assume unit mass and thermalize to Maxwell-Boltzmann distribution
            snap.particles.mass[:] = 1.0
            vel = np.random.normal(scale=np.sqrt(sim.ensemble.kT))
            snap.particles.velocity[:] = vel-np.mean(vel,axis=1)

            # read snapshot
            sim.system = hoomd.init.read_snapshot(snap)

        self.attach_potentials(sim)

## integrators

class AddIntegrator(simulate.AddIntegrator):
    def __init__(self, dt):
        super().__init__(dt)

    def attach_integrator(self, sim):
        with sim.context:
            hoomd.md.integrate.mode_standard(self.dt)

class AddBrownianIntegrator(AddIntegrator):
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
            ld = hoomd.md.integrate.brownian(group=all_,
                                             kT=sim.ensemble.kT,
                                             seed=seed,
                                             **self.options)
            for t in sim.ensemble:
                try:
                    gamma = self.friction[t]
                except TypeError:
                    gamma = self.friction
                ld.set_gamma(t,gamma)

class AddLangevinIntegrator(AddIntegrator):
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
            ld = hoomd.md.integrate.langevin(group=all_,
                                             kT=sim.ensemble.kT,
                                             seed=seed,
                                             **self.options)
            for t in sim.ensemble:
                try:
                    gamma = self.friction[t]
                except TypeError:
                    gamma = self.friction
                ld.set_gamma(t,gamma)

class AddNPTIntegrator(AddIntegrator):
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
            hoomd.md.integrate.npt(group=all_,
                                   kT=sim.ensemble.kT,
                                   tau=self.tau_T,
                                   P=sim.ensemble.P,
                                   tauP=self.tau_P,
                                   **self.options)

class AddNVTIntegrator(AddIntegrator):
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
            hoomd.md.integrate.nvt(group=all_,
                                   kT=sim.ensemble.kT,
                                   tau=self.tau_T,
                                   **self.options)

class Run(simulate.Run):
    """Advance the simulation."""
    def __call__(self, sim):
        with sim.context:
            hoomd.run(self.steps)

## analyzers

class AddEnsembleAnalyzer(simulate.AddAnalyzer):
    def __init__(self, every):
        super().__init__(self, every)
        self.recorder = None

    class StateRecorder:
        def __init__(self, sim, logger):
            self.sim = sim
            self.logger = logger
            self.P = []
            self.V = []

        def __call__(self, timestep):
            self.P.append(self.logger.query('pressure'))
            self.V.append(self.logger.query('volume'))

    def __call__(self, sim):
        with sim.context:
            # thermodynamic properties
            logger = hoomd.analyze.log(filename=None,
                                       quantities=['pressure','volume'],
                                       period=self.every)
            self.recorder = StateRecorder(sim,logger)
            hoomd.analyze.callback(callback=self.recorder,period=self.every)

    @property
    def ensemble(self):
        sim = self.recorder.sim
        ens = sim.ensemble.copy()
        ens.clear()

        if not ens.constant('P'):
            ens.P = np.mean(self.recorder.P)
        if not ens.constant('V'):
            ens.V = np.mean(self.recorder.V)

        return ens
