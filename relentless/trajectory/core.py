__all__ = ['Box','Snapshot','Trajectory']

import numpy as np

class Box(object):
    def __init__(self, L, periodic=True):
        # deduce the box
        L = np.asarray(np.atleast_1d(L))
        if L.shape == (1,):
            self._lo = np.zeros(3, dtype=L.dtype)
            self._hi = np.full(3, L)
        elif L.shape == (3,):
            self._lo = np.zeros(3, dtype=L.dtype)
            self._hi = L.copy()
        elif L.shape == (2,3):
            self._lo = L[0].copy()
            self._hi = L[1].copy()
        else:
            raise TypeError('L must be either a scalar, a 3-tuple, or two 3-tuples.')

        periodic = np.array(np.atleast_1d(periodic), dtype=bool)
        if periodic.shape == (1,):
            self._periodic = np.full(3, periodic, dtype=bool)
        elif periodic.shape == (3,):
            self._periodic = periodic
        else:
            raise TypeError('periodic must be a bool or a 3-tuple')

        self._L = self._hi - self._lo
        self._volume = np.prod(self._L)

    @property
    def hi(self):
        return self._hi

    @property
    def lo(self):
        return self._lo

    @property
    def L(self):
        return self._L

    @property
    def periodic(self):
        return self._periodic

    @property
    def volume(self):
        return self._volume

    def wrap(self, r):
        img = np.array(np.atleast_1d(r), dtype=np.float64)
        nimg = np.zeros(img.shape, dtype=np.int32)

        p = self.periodic
        nimg[...,p] = np.floor((img[...,p]-self.lo[p])/self.L[p]).astype(np.int32)
        img[...,p] -= nimg[...,p]*self.L[p]
        return img,nimg

    def min_image(self, dr):
        img = np.array(np.atleast_1d(dr), dtype=np.float64)
        p = self.periodic
        img[...,p] -= np.round(img[...,p]/self.L[p])*self.L[p]
        return img

class Snapshot(object):
    def __init__(self, N, box=None):
        self._N = N
        self.timestep = 0
        self.box = box
        self._positions = None
        self._types = None

    @property
    def N(self):
        return self._N

    @property
    def positions(self):
        if self._positions is None:
            self._positions = np.empty((self.N,3), dtype=np.float64)
        return self._positions

    @positions.setter
    def positions(self, value):
        if self._positions is None:
            value = np.array(value, dtype=np.float64)
            assert value.shape == (self.N,3), "Positions must be an Nx3 array."
            self._positions = value
        else:
            np.copyto(self._positions, value)

    @property
    def types(self):
        if self._types is None:
            self._types = np.empty(self.N, dtype='<U2')
        return self._types

    @types.setter
    def types(self, value):
        if self._types is None:
            value = np.array(value)
            assert value.shape == (self.N,), "Positions must be a length N array."
            self._types = value
        else:
            np.copyto(self._types, value)

class Trajectory(object):
    def __init__(self, snapshots=[]):
        self._snapshots = snapshots

    def append(self, snapshot):
        self._snapshots.append(snapshot)

    def extend(self, snapshots):
        self._snapshots.extend(snapshots)

    def __getitem__(self, key):
        return self._snapshots[key]

    def __len__(self):
        return len(self._snapshots)

    def __iter__(self):
        return iter(self._snapshots)

    def __next__(self):
        return next(self._snapshots)

    def sort(self):
        self._snapshots.sort(key=lambda s : s.timestep)

class TrajectoryReader(object):
    def __init__(self, filename):
        self.filename = filename

    def load(self, env, step):
        raise NotImplementedError()
