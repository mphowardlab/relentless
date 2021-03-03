"""Core environment functionalities.

Todo
----
Improve this documentation for developers!

"""
__all__ = ['Environment','Policy']

import subprocess

from relentless.data import Directory

class Policy:
    """Execution policy.

    Specifies the execution policy for commands dispatched to the shell. This
    can be useful for controlling, e.g., the number of MPI ranks for a program
    using spatial domain decomposition or the number of OpenMP threads consumed
    by a multithreaded analysis program, in a system-independent manner.

    No error checking is done to ensure that sufficient resources are available
    for the execution policy, so oversubscription can occur if values are
    chosen incorrectly. It is the caller's responsibility to use sensible
    values that optimize performance for their problem and resources.

    Parameters
    ----------
    procs : int or None
        Number of MPI ranks ("processors") to use for execution.

    threads : int or None
        Number of OpenMP threads to use for execution.

    Examples
    --------
    Policy for an MPI application on 8 processors::

        p = Policy(procs=8)

    Policy for an OpenMP application using 8 threads::

        p = Policy(threads=8)

    Policy for an MPI+OpenMP application using 4 processors with 2 threads
    each::

        p = Policy(procs=4, threads=2)

    See also
    --------
    :class:`Environment`

    Notes
    -----
    Specified parameters must have integer values of at least 1. Unspecified
    parameters default to a choice by the :class:`Environment`, which will
    usually be determined by environment variables or the system configuration.
    It is better to be explicit!

    """
    def __init__(self, procs=None, threads=None):
        if procs is not None:
            procs = int(procs)
            if not procs >= 1:
                raise ValueError('Number of processors must be >= 1.')
            else:
                self.procs = procs
        else:
            self.procs = None

        if threads is not None:
            threads = int(threads)
            if not threads >= 1:
                raise ValueError('Threads must be >= 1.')
            else:
                self.threads = threads
        else:
            self.threads = None

class Environment:
    """Execution environment.

    Defines an environment for executing commands in a `project` using a given
    policy. The environment has a `path`, which defines the filesystem (base
    directory) for the `project` and defaults to the current working directory.
    Within a `project`, there is `data` associated with a given `step`.

    An environment is intended to be easily inheritable for system
    specialization. This can be as simple as redefining the properties
    `mpiexec` and `always_wrap`.

    Parameters
    ----------
    path : str
        Filesystem path for the `project`, defaults to the current directory.

    mock : bool
        If `True`, command strings are processed but **not** executed. This can
        be useful for conducting a dry run of a new algorithm if running
        the commands takes significant time.

    Examples
    --------
    Defining a new environment class::

        class OpenMPI(relentless.Environment):
            @property
            def mpiexec(self):
                return 'mpirun -n {np}'

            @property
            def always_wrap(self):
                return False

    Creating a project environment::

        env = OpenMPI(path='workspace')

    Mocking commands::

        env = OpenMPI(path='workspace', mock=True)
        env.call('sleep 10h', Policy())
        # returns immediately!

    See also
    --------
    py:class:`Policy`

    Notes
    -----
    The `step` data schema for the `project` is consistent with iterative
    optimization but may be relaxed or modified in future.

    """
    always_wrap = False

    def __init__(self, path='.', mock=False):
        self._project = Directory(path)
        self.mock = mock

    def call(self, cmd, policy):
        """Execute a command through the shell.

        The environment interprets a :class:`Policy` to create and execute a
        command through the shell. Currently, this involves wrapping `cmd` with
        environment-specific MPI and OpenMP commands and flags set by `policy`.

        Parameters
        ----------
        cmd : str or list or tuple
            Command to execute in the shell.
        policy : :class:`Policy`
            Execution policy for the command.

        Returns
        -------
        str
            The command that was executed through the shell.

        Examples
        --------
        Calling a shell command::

            env = Environment()
            env.call('echo "Hello world!" >> out.log', policy=Policy())

        Calling a command with MPI::

            env.call('echo "Hello world!" >> out.log', policy=Policy(procs=2))

        Calling a command using a list::

            env.call(['ls','-l'], policy=Policy())

        Raises
        ------
        RuntimeError
            If the command is executed and returns a non-zero exit code.

        """
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
            task = subprocess.Popen(cmd, shell=True)
            output = task.communicate()
            if task.returncode != 0:
                raise RuntimeError('Shell subprocess returned non-zero exit code.')

        return cmd

    @property
    def mpiexec(self):
        """str or None Format string for MPI-enabled policies.

        The string is the MPI wrapper command and can accept the `{np}`
        format variable to set the number of processors. This attribute is
        `None` for environments without MPI.
        """
        # TODO: This should probably be a method accepting parameters to work
        # with GPUs and hybrid applications.
        return None

    @property
    def always_wrap(self):
        """bool If `True`, all commands use the MPI wrapper.

        This may be required for systems where compute-intensive processes
        may not be executed on the node running the python interpreter, e.g.,
        a PBS mom node. The default value is False.
        """
        return False

    @property
    def project(self):
        """:class:`Directory` The base directory for the project."""
        return self._project

    def data(self, step):
        """Access a data directory for the project.

        Parameters
        ----------
        step : int
            Counter referring to the current step in the optimization.

        Returns
        -------
        :class:`Directory`
            The data directory associated with a `step`.

        """
        return self.project.directory(str(step))
