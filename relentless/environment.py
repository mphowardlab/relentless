from __future__ import print_function
import os
import subprocess

from . import utils

class Policy(object):
    def __init__(self, procs=None, threads=None):
        if procs is not None:
            if not procs >= 1:
                raise ValueError('Number of processors must be >= 1.')
            else:
                self.procs = int(procs)
        else:
            self.procs = None

        if threads is not None:
            if not threads >= 1:
                raise ValueError('Threads must be >= 1.')
            else:
                self.threads = int(threads)
        else:
            self.threads = None

class Environment(object):
    mpiexec = None
    always_wrap = False

    def __init__(self, scratch, work, archive=False, mock=False):
        self._scratch = scratch
        self._work = work
        self.cwd = None

        self.archive = archive
        self.mock = mock

    def reset(self):
        self.cwd = None

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.reset()
        print('Cleanup scratch? {}'.format(self.archive))

    def call(self, cmd, policy):
        # OpenMP threading
        if policy.threads is not None:
            omp = 'OMP_NUM_THREADS={}'.format(policy.threads)
        else:
            omp = ''

        # MPI wrapping
        if policy.procs is not None:
            if self.mpiexec is None:
                raise ValueError('Cannot launch MPI task without MPI executable.')
            mpi = self.mpiexec.format(np=policy.procs)
        elif self.always_wrap:
            assert self.mpiexec, 'MPI wrapping is configured but there is no MPI executable.'
            mpi = self.mpiexec.format(np=1)
        else:
            mpi = ''

        # turn a list of commands into a string if supplied
        if not utils.isstr(cmd):
            cmd = ' '.join(cmd)

        cmd = '{omp} {mpi} {cmd}'.format(omp=omp,mpi=mpi,cmd=cmd).strip()
        if not self.mock:
            subprocess.Popen(cmd, shell=True).communicate()
        else:
            print(cmd)

    def scratch(self, path=None):
        if path is not None:
            return os.path.join(self._scratch, path)
        else:
            return self._scratch

    def work(self, path=None):
        if path is not None:
            return os.path.join(self._work, path)
        else:
            return self._work

class SLURM(Environment):
    mpiexec = 'srun'
    always_wrap = False

    def __init__(self, scratch, work, archive=False, mock=False):
        super(SLURM, self).__init__(scratch, work, archive, mock)

class Lonestar(Environment):
    mpiexec = 'ibrun'
    always_wrap = False

    def __init__(self, scratch=None, work=None, archive=False, mock=False):
        super(Lonestar, self).__init__(scratch,work, archive, mock)

class Stampede2(Environment):
    mpiexec = 'ibrun'
    always_wrap = False

    def __init__(self, scratch=None, work=None, archive=False, mock=False):
        super(Stampede2, self).__init__(scratch,work, archive, mock)
