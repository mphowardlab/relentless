import os
import numpy as np
try:
    from mpi4py import MPI
    _has_mpi4py = True
except ImportError:
    _has_mpi4py = False

class Communicator:
    def __init__(self, comm=None, root=0):
        if _has_mpi4py:
            if comm is None:
                self._comm = MPI.COMM_WORLD
            else:
                self._comm = comm
            self._comm.Set_errhandler(MPI.ERRORS_ARE_FATAL)
        else:
            hints = ('MV2_COMM_WORLD_LOCAL_RANK',
                     'OMPI_COMM_WORLD_RANK',
                     'PMI_RANK',
                     'ALPS_APP_PE')
            if any(h in os.environ for h in hints):
                raise RuntimeError('Python seems to running in MPI, but mpi4py is not installed.')
            self._comm = None

        if self.enabled:
            self._size = self.comm.Get_size()
            self._rank = self.comm.Get_rank()
            self._root = root
        else:
            self._size = 1
            self._rank = 0
            self._root = 0

        if self.root < 0 or self.root >= self.size:
            raise ValueError('Root rank ID out of bounds.')

    @property
    def comm(self):
        return self._comm

    @property
    def enabled(self):
        return self._comm is not None

    @property
    def size(self):
        return self._size

    @property
    def rank(self):
        return self._rank

    @property
    def root(self):
        return self._root

    def bcast(self, data, root=None):
        if not self.enabled or self.size == 1:
            return data

        if root is None:
            root = self.root

        return self.comm.bcast(data,root)

    def bcast_numpy(self, data, root=None):
        if not self.enabled or self.size == 1:
            return data

        if root is None:
            root = self.root

        # broadcast the shape and data type
        if self.rank == root:
            shape = data.shape
            dtype = data.dtype
        else:
            shape = None
            dtype = None
        shape = self.bcast(shape,root)
        dtype = self.bcast(dtype,root)

        # allocate memory if needed (storage may already exist)
        if self.rank != root:
            try:
                alloc = data.shape != shape or data.dtype != dtype
            except AttributeError:
                alloc = True
            if alloc:
                data = np.empty(shape,dtype=dtype)

        # broadcast from the root rank
        self.comm.Bcast(data,root)

        return data

    def loadtxt(self, filename, root=None, **kwargs):
        if root is None:
            root = self.root

        # load on root rank and broadcast
        if self.rank == root:
            dat = np.loadtxt(filename, **kwargs)
        else:
            dat = None
        self.bcast_numpy(dat,root)

        return dat

world = Communicator()
