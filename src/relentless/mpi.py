"""
====================================
MPI communication (`relentless.mpi`)
====================================

.. currentmodule:: relentless.mpi

Parallel execution using the Message Passing Interface (MPI) is supported.
Typically, the calculations required to steer an optimization are simple, so the
primary reason to support MPI is for :class:`~relentless.simulate.simulate.Simulation`
engines that will run on multiple processes. In Python, the MPI interface is
accessed using the :mod:`mpi4py` package.

MPI processes run within a :class:`Communicator`. The default communicator is
the MPI "world," which is all running processes. You can create your own
:class:`Communicator` using subsets of these processes if you do not want to
use all of them.

All commands in your script are intended to be *collective* within a communicator,
meaning that they should be executed by all processes. Behind the scenes, the
appropriate MPI collectives will be invoked to keep things synchronized. Then,
running the script under MPI can be as simple as invoking it with the MPI launcher
on your system::

    mpirun python3 script.py

Some caution must be taken with opening file handles when running under MPI. If
you do not take special steps, each process will acquire a handle to the same
resource. This can quickly overwhelm the operating-system limit. Best practice
for small files is to allow one rank to load the file, then broadcast it to the
other ranks. A convenience wrapper is given in :meth:`Communicator.loadtxt`;
see its documentation for more details.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    Communicator

"""

import json
import os

import numpy

_mpi_env_hints = (
    "MV2_COMM_WORLD_LOCAL_RANK",
    "OMPI_COMM_WORLD_RANK",
    "PMI_RANK",
    "ALPS_APP_PE",
)
_mpi_running = any(h in os.environ for h in _mpi_env_hints)
if _mpi_running:
    try:
        from mpi4py import MPI
    except ImportError as exc:
        raise ImportError(
            "relentless seems to running in MPI, but mpi4py is not installed."
        ) from exc


class Communicator:
    """MPI communicator.

    Wrapper around an MPI communicator. Methods will gracefully degrade to
    single-processor functions if not running under MPI or there is only a single
    rank in the communicator.

    Parameters
    ----------
    comm : :class:`mpi4py.MPI.Comm`
        Communicator backend. Defaults to ``None``, which is converted to
        :data:`mpi4py.MPI.COMM_WORLD` if running under MPI.
    root : int
        Root rank index (defaults to 0).

    Raises
    ------
    IndexError
        If the ``root`` rank does not lie within the valid range for the communicator.

    """

    def __init__(self, comm=None, root=0):
        if _mpi_running:
            if comm is None:
                self._comm = MPI.COMM_WORLD
            else:
                self._comm = comm
            self._comm.Set_errhandler(MPI.ERRORS_ARE_FATAL)
        else:
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
            raise IndexError("Root rank ID out of bounds.")

    @property
    def comm(self):
        """:class:`mpi4py.MPI.Comm` MPI communicator.

        This attribute is ``None`` if running on a single processor.

        """
        return self._comm

    @property
    def enabled(self):
        """bool: True if MPI communication is enabled."""
        return self._comm is not None

    @property
    def size(self):
        """int: Number of ranks in the MPI communicator."""
        return self._size

    @property
    def rank(self):
        """int: Index of rank in the MPI communicator."""
        return self._rank

    @property
    def root(self):
        """int: Index of the root rank in the MPI communicator."""
        return self._root

    @property
    def rank_is_root(self):
        """bool: True if the MPI rank is the root rank."""
        return self.rank == self.root

    def barrier(self):
        """Create barrier for all ranks in the MPI communicator."""
        if self.enabled and self.size > 1:
            self.comm.barrier()

    def bcast(self, data, root=None):
        """Broadcast Python object to all ranks.

        Broadcasting is a one-to-all communication pattern that can be used
        to synchronize data across ranks. This method wraps around the
        :meth:`mpi4py.MPI.Comm.bcast` method, which only operates on Python
        data types that are picklable. If you need to broadcast a NumPy array
        that uses the Python buffer protocol, you should use :meth:`bcast_numpy`.

        Parameters
        ----------
        data : object
            Any Python picklable object.
        root : int
            Rank to broadcast from. If ``None`` (default), broadcast from the
            :attr:`root` of the communicator.

        Returns
        -------
        object
            The broadcast data on all ranks.

        Example
        -------
        Broadcast an integer:

        .. code::

            comm = relentless.mpi.Communicator()
            if comm.rank_is_root:
                x = 42
            else:
                x = None
            x = comm.bcast(x)

        """
        if not self.enabled or self.size == 1:
            return data

        if root is None:
            root = self.root

        return self.comm.bcast(data, root)

    def bcast_numpy(self, data, root=None):
        """Broadcast NumPy array to all ranks.

        Broadcasting is a one-to-all communication pattern that can be used
        to synchronize data across ranks. This method wraps around the
        :meth:`mpi4py.MPI.Comm.Bcast` method for NumPy arrays. The method will
        ensure that ``data`` has the correct size and type on all ranks by
        first broadcasting this information; if ``data`` is not allocated in
        this way, it will be allocated automatically.

        Parameters
        ----------
        data : :class:`numpy.ndarray`
            NumPy array to broadcast.
        root : int
            Rank to broadcast from. If ``None`` (default), broadcast from the
            :attr:`root` of the communicator.

        Returns
        -------
        :class:`numpy.ndarray`
            The broadcast array on all ranks.

        Examples
        --------
        Broadcast an array:

        .. code::

            comm = relentless.mpi.Communicator()
            if comm.rank_is_root:
                x = numpy.array([1,2,3], dtype=numpy.int32)
            else:
                x = None
            x = comm.bcast_numpy(x)

        Broadcast an array with pre-existing storage:

        .. code::

            comm = relentless.mpi.Communicator()
            if comm.rank_is_root:
                x = numpy.array([1,2,3], dtype=numpy.int32)
            else:
                x = numpy.empty(3, dtype=numpy.int32)
            x = comm.bcast_numpy(x)

        """
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
        shape = self.bcast(shape, root)
        dtype = self.bcast(dtype, root)

        # allocate memory if needed (storage may already exist)
        if self.rank != root:
            try:
                alloc = data.shape != shape or data.dtype != dtype
            except AttributeError:
                alloc = True
            if alloc:
                data = numpy.empty(shape, dtype=dtype)

        # broadcast from the root rank
        self.comm.Bcast(data, root)

        return data

    def loadtxt(self, filename, root=None, **kwargs):
        """Load text from a file and broadcast.

        Data is loaded from file using :func:`numpy.loadtxt` on the ``root`` rank,
        then broadcast to all ranks. This function can be called with essentially
        no overhead in single-processor calculations, but prevents oversubscribing
        file handles when running under MPI.

        Parameters
        ----------
        filename : str
            Name of the file to load.
        root : int
            Rank to broadcast from. If ``None`` (default), broadcast from the
            :attr:`root` of the communicator.
        **kwargs
            Optional keyword arguments to :func:`numpy.loadtxt`.

        Returns
        -------
        :class:`numpy.ndarray`
            The data from ``filename``.

        Example
        -------
        Load a file:

        .. code::

            comm = relentless.mpi.Communicator()
            dat = comm.loadtxt("gr.dat")

        """
        if root is None:
            root = self.root

        # load on root rank and broadcast
        if self.rank == root:
            dat = numpy.loadtxt(filename, **kwargs)
        else:
            dat = None
        dat = self.bcast_numpy(dat, root)

        return dat

    def load_json(self, filename, root=None):
        """Load a JSON file and broadcast.

        Data is loaded from file using :func:`json.load` on the ``root`` rank,
        then broadcast to all ranks. This function can be called with essentially
        no overhead in single-processor calculations, but prevents oversubscribing
        file handles when running under MPI.

        Parameters
        ----------
        filename : str
            Name of the file to load.
        root : int
            Rank to broadcast from. If ``None`` (default), broadcast from the
            :attr:`root` of the communicator.

        Returns
        -------
        dict
            The data from ``filename``.

        Example
        -------
        Load a file:

        .. code::

            comm = relentless.mpi.Communicator()
            dat = comm.load_json("ensemble.json")


        """
        if root is None:
            root = self.root

        if self.rank == root:
            with open(filename) as f:
                dat = json.load(f)
        else:
            dat = None
        dat = self.bcast(dat, root)

        return dat


world = Communicator()
