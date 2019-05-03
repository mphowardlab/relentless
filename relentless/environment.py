from __future__ import print_function
import os
import subprocess

from . import utils

class Environment(object):
    mpiexec = None
    always_wrap = False

    def __init__(self, scratch, work, mock=False):
        self._scratch = scratch
        self._work = work
        self.mock = mock

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass

    def call(self, cmd, procs=None, threads=None):
        # OpenMP threading
        if threads is not None:
            if not threads >= 1:
                raise ValueError("Threads must be >= 1.")
            omp = "OMP_NUM_THREADS={}".format(int(threads))
        else:
            omp = ""

        # MPI wrapping
        if procs is not None:
            if self.mpiexec is None:
                raise ValueError('Cannot launch MPI task without MPI executable.')
            if not procs >= 1:
                raise ValueError("Number of processors must be >= 1.")
            mpi = self.mpiexec.format(np=int(procs))
        elif self.always_wrap:
            assert self.mpiexec, "MPI wrapping is configured but there is no MPI executable."
            mpi = self.mpiexec.format(np=1)
        else:
            mpi = ""

        # turn a list of commands into a string if supplied
        if not utils.isstr(cmd):
            cmd = ' '.join(cmd)

        cmd = "{omp} {mpi} {cmd}".format(omp=omp,mpi=mpi,cmd=cmd).strip()
        if not self.mock:
            subprocess.Popen(cmd, shell=True).communicate()
        else:
            print(cmd)

    def scratch(self, filename=None):
        if filename is not None:
            return os.path.join(self._scratch, filename)
        else:
            return self._scratch

    def work(self, filename=None):
        if filename is not None:
            return os.path.join(self._work, filename)
        else:
            return self._work

class SLURM(Environment):
    mpiexec = 'srun'
    always_wrap = False

    def __init__(self, scratch, work, mock=False):
        super(SLURM, self).__init__(scratch, work, mock)

class Lonestar(Environment):
    mpiexec = 'ibrun'
    always_wrap = False

    def __init__(self, scratch=None, work=None, mock=False):
        super(Lonestar, self).__init__(scratch,work, mock)

class Stampede2(Environment):
    mpiexec = 'ibrun'
    always_wrap = False

    def __init__(self, scratch=None, work=None, mock=False):
        super(Stampede2, self).__init__(scratch,work, mock)
