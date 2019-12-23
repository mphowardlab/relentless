"""Generic runtime environments.

The following systems are currently supported:

* :py:class:`OpenMPI`
* :py:class:`SLURM`

"""
from . import core

class OpenMPI(core.Environment):
    """OpenMPI environment.

    This generic environment supports the OpenMPI library and launcher.
    """

    @property
    def mpiexec(self):
        """str Format string for MPI-enabled policies.

        Commands are wrapped in the `mpirun` launcher::

            mpirun -n {np}

        """
        return 'mpirun -n {np}'

    @property
    def always_wrap(self):
        """bool False

        Generic systems do not wrap.

        """
        return False

class SLURM(core.Environment):
    """SLURM environment.

    This generic environment supports the SLURM scheduler and its MPI wrapper.

    """

    @property
    def mpiexec(self):
        """str Format string for MPI-enabled policies.

        Commands are wrapped in the `srun` launcher::

            srun

        """
        # TODO: support proc counts
        return 'srun'

    @property
    def always_wrap(self):
        """bool False

        Generic systems do not wrap.

        """
        return False
