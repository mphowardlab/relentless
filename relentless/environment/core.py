__all__ = ['Environment','Policy']

import os
import subprocess

class TemporaryDirectory(object):
    """ Temporary working directory.
    """
    def __init__(self, path):
        self.path = os.path.abspath(path)
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def __enter__(self):
        self.start = os.getcwd()
        os.chdir(self.path)
        return self.path

    def __exit__(self, exception_type, exception_value, traceback):
        os.chdir(self.start)

class Policy(object):
    """ Execution poliy."""
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

    def __init__(self, path, mock=False):
        self._path = os.path.abspath(path)
        self.mock = mock

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass

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
        if not isinstance(cmd, str):
            cmd = ' '.join(cmd)

        cmd = '{omp} {mpi} {cmd}'.format(omp=omp,mpi=mpi,cmd=cmd).strip()
        if not self.mock:
            subprocess.Popen(cmd, shell=True).communicate()

        return cmd

    @property
    def project(self):
        return TemporaryDirectory(self._path)

    def data(self, step):
        return TemporaryDirectory(os.path.join(self._path, str(step)))
