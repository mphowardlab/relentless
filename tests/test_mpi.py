"""Unit tests for mpi module."""

import os
import tempfile
import unittest

import numpy

import relentless

try:
    from mpi4py import MPI
except ImportError:
    pass

has_mpi = relentless.mpi._mpi_running


class test_Communicator(unittest.TestCase):
    def setUp(self):
        self.comm = relentless.mpi.world

    def test_init(self):
        if has_mpi:
            self.assertTrue(self.comm.comm is MPI.COMM_WORLD)
            self.assertTrue(self.comm.enabled)
            self.assertEqual(self.comm.size, MPI.COMM_WORLD.Get_size())
            self.assertEqual(self.comm.rank, MPI.COMM_WORLD.Get_rank())
            self.assertEqual(self.comm.root, 0)
        else:
            self.assertTrue(self.comm.comm is None)
            self.assertFalse(self.comm.enabled)
            self.assertEqual(self.comm.size, 1)
            self.assertEqual(self.comm.rank, 0)
            self.assertEqual(self.comm.root, 0)

    @unittest.skipUnless(has_mpi, "Needs MPI")
    def test_split(self):
        world = MPI.COMM_WORLD.Dup()
        self.assertTrue(world is not MPI.COMM_WORLD)

        comm = relentless.mpi.Communicator(world)
        self.assertTrue(comm is not relentless.mpi.world)

    def test_bcast(self):
        # broadcast from default root
        if self.comm.rank_is_root:
            x = 42
        else:
            x = None
        x = self.comm.bcast(x)
        self.assertEqual(x, 42)

        # broadcast from specified root
        if self.comm.size >= 2:
            root = 1
            if self.comm.rank == root:
                x = 7
            else:
                x = None
            x = self.comm.bcast(x, root=root)
            self.assertEqual(x, 7)

    def test_bcast_numpy(self):
        # float array
        if self.comm.rank_is_root:
            x = numpy.array([1.0, 2.0], dtype=numpy.float64)
        else:
            x = None
        x = self.comm.bcast_numpy(x)
        self.assertEqual(x.shape, (2,))
        self.assertEqual(x.dtype, numpy.float64)
        numpy.testing.assert_allclose(x, [1.0, 2.0])

        # float array from specified root
        if self.comm.size >= 2:
            root = 1
            if self.comm.rank == root:
                x = numpy.array([3.0, 4.0], dtype=numpy.float64)
            else:
                x = None
            x = self.comm.bcast_numpy(x, root=root)
            self.assertEqual(x.shape, (2,))
            self.assertEqual(x.dtype, numpy.float64)
            numpy.testing.assert_allclose(x, [3.0, 4.0])

        # prealloc'd float array
        if self.comm.rank_is_root:
            x = numpy.array([5.0, 6.0], dtype=numpy.float64)
        else:
            x = numpy.zeros((2,), dtype=numpy.float64)
        y = self.comm.bcast_numpy(x)
        self.assertTrue(y is x)
        self.assertEqual(y.shape, (2,))
        self.assertEqual(y.dtype, numpy.float64)
        numpy.testing.assert_allclose(y, [5.0, 6.0])

        # incorrectly alloc'd float array
        if self.comm.rank_is_root:
            x = numpy.array([5.0, 6.0], dtype=numpy.float64)
        else:
            x = numpy.zeros((2,), dtype=numpy.int32)
        y = self.comm.bcast_numpy(x)
        if self.comm.rank_is_root:
            self.assertTrue(y is x)
        else:
            self.assertTrue(y is not x)
        self.assertEqual(y.shape, (2,))
        self.assertEqual(y.dtype, numpy.float64)
        numpy.testing.assert_allclose(y, [5.0, 6.0])

    def test_loadtxt(self):
        # create file
        if self.comm.rank_is_root:
            tmp = tempfile.NamedTemporaryFile(delete=False)
            tmp.write(b"1 2\n 3 4\n")
            tmp.close()
            filename = tmp.name
        else:
            filename = None
        filename = self.comm.bcast(filename)

        # load data
        dat = self.comm.loadtxt(filename)

        # unlink before testing in case an exception gets raised
        if self.comm.rank_is_root:
            os.unlink(tmp.name)

        numpy.testing.assert_allclose(dat, [[1.0, 2.0], [3.0, 4.0]])


if __name__ == "__main__":
    unittest.main()
