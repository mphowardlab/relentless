import itertools

import numpy
import scipy.spatial

from relentless.model import extent
from . import simulate
from relentless.model import variable

## initializers
class InitializeFromFile(simulate.GenericOperation):
    """Initialize a simulation from a file.

    Description.

    Parameters
    ----------
    filename : str
        Initial configuration.

    """
    def __init__(self, filename):
        super().__init__(filename)

class InitializeRandomly(simulate.GenericOperation):
    """Initialize a randomly generated simulation box.

    If ``diameters`` is ``None``, the particles are randomly placed in the box.
    This can work pretty well for low densities, particularly if
    :class:`~relentless.simulate.MinimizeEnergy` is used to remove overlaps
    before starting to run a simulation. However, it will typically fail for
    higher densities, where there are many overlaps that are hard to resolve.

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
        are randomly inserted without checking their sizes. The value of a
        diameter can be a :class:`~relentless.variable.Variable`, which will be
        evaluated at the time the operation is called.

    """
    def __init__(self, seed, N, V, T=None, masses=None, diameters=None):
        super().__init__(seed, N, V, T, masses, diameters)

    @staticmethod
    def _make_orthorhombic(V):
        # get the orthorhombic bounding box
        if isinstance(V, extent.TriclinicBox):
            Lx,Ly,Lz,xy,xz,yz = V.as_array('HOOMD')
            aabb = numpy.array([Lx/numpy.sqrt(1.+xy**2+(xy*yz-xz)**2), Ly/numpy.sqrt(1.+yz**2), Lz])
        elif isinstance(V, extent.ObliqueArea):
            Lx,Ly,xy = V.as_array('HOOMD')
            aabb = numpy.array([Lx/numpy.sqrt(1.+xy**2), Ly])
        else:
            raise TypeError('Random initialization currently only supported in triclinic/oblique extents')
        return aabb

    @staticmethod
    def _random_particles(seed, N, V):
        rng = numpy.random.default_rng(seed)
        aabb = InitializeRandomly._make_orthorhombic(V)

        positions = aabb*rng.uniform(size=(sum(N.values()), len(aabb)))
        positions = V.wrap(positions)

        types = []
        for i,Ni in N.items():
            types.extend([i]*Ni)

        return positions, types

    @staticmethod
    def _pack_particles(seed, N, V, diameters):
        rng = numpy.random.default_rng(seed)
        aabb = InitializeRandomly._make_orthorhombic(V)
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
            if isinstance(di, variable.Variable):
                di = di.value
            if dimension == 3:
                # fcc lattice
                a = numpy.sqrt(2.)*di
                lattice = numpy.array([a,a,a])
                cell_coord = numpy.array([[0.,0.,0.],[0.5,0.5,0.],[0.5,0.,0.5],[0.,0.5,0.5]])
            elif dimension == 2:
                a = di
                b = numpy.sqrt(3.)*di
                lattice = numpy.array([a,b])
                cell_coord = numpy.array([[0.,0.],[0.5,0.5]])
            else:
                raise ValueError('Only 3d and 2d packings are supported')
            # this part generates a cartesian mesh of unit cells that fit within a box, such that no particle
            # can cross the outside of the aabb. then, it loops through all the cells and puts the particles in
            # place. everything is based on fractional coordinates, so it gets scaled by the lattice.
            num_lattice = numpy.floor((aabb-di)/lattice).astype(int)
            sites = numpy.zeros((numpy.prod(num_lattice)*cell_coord.shape[0], dimension), dtype=numpy.float64)
            first = 0
            for cell_origin in itertools.product(*[numpy.arange(n) for n in num_lattice]):
                sites[first:first+cell_coord.shape[0]] = lattice*(cell_origin + cell_coord)
                first += cell_coord.shape[0]
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
