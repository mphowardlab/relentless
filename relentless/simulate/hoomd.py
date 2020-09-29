__all__ = ['HOOMD']

from packaging import version

import numpy as np

from relentless.core.collections import PairMatrix
from relentless.core.volume import TriclinicBox
from .simulate import Simulation,SimulationInstance

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

class HOOMD(Simulation):
    default_options = {'dt':0.005, 'r_buff':0.4}

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

    def initialize(self, sim):
        sim.context = hoomd.SimulationContext()

    def analyze(self, sim):
        pass

class HOOMDInitializer:
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
            r = potentials[i,j]['r']
            u = potentials[i,j]['u']
            f = potentials[i,j]['f']

            # validate table size
            if table_size is None:
                table_size = len(r)
            if len(r) != table_size or len(u) != table_size or len(f) != table_size:
                raise ValueError('HOOMD requires equal sized tables.')

            files[i,j] = 'table_{i}_{j}.dat'.format(i=i,j=j)
            header = '# Tabulated pair for ({i},{j})\n'.format(i=i,j=j)
            header += '# r u f'
            np.savetxt(files[i,j],
                       np.column_stack((r,u,f)),
                       header=header,
                       comments='')

        # create potentials in HOOMD script
        with sim.context:
            self.nl = hoomd.md.nlist.tree(r_buff=sim.r_buff)
            self.pair = hoomd.md.pair.table(width=table_size, nlist=self.nl)
            for i,j in files:
                self.pair.set_from_file(i,j,files[i,j])

class HOOMDInitializeRandom(HOOMDInitializer):
    def __call__(self, sim):
        with sim.context:
            snap,box = self.make_snapshot(sim)

            # randomly place particles in fractional coordinates
            rs = np.random.uniform(size=(snap.particles.N,3))
            snap.particles.position[:] = box.make_absolute(rs)

            # set types of each
            snap.particles.types = tuple(sim.ensemble.types)
            snap.particles.typeid[:] = np.repeat(np.arange(len(sim.ensemble.types)),
                                                [sim.ensemble.N[t] for t in sim.ensemble.types])

            # read snapshot
            sim.system = hoomd.init.read_snapshot(snap)

        self.attach_potentials(sim)
