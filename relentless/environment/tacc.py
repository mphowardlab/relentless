"""Runtime environments at the Texas Advanced Computing Center (TACC).

The following systems are currently supported:

* :py:class:`Lonestar5` (`ls5.tacc.utexas.edu`)
* :py:class:`Stampede2` (`stampede2.tacc.utexas.edu`)

"""
from . import core

class Lonestar5(core.Environment):
    """Lonestar5 environment.

    This environment supports Lonestar5 at TACC (`ls5.tacc.utexas.edu`). `ls5`
    is a hybrid compute cluster having both 16-core CPU nodes and NVIDIA K20 GPU
    nodes. It is recommended to define policies having no more than 16 **total**
    compute threads per node. GPUs are currently not officially supported by
    :py:class:`~.core.Policy`, but may be experimentally used in certain
    configurations.

    """

    @property
    def mpiexec(self):
        """str Format string for MPI-enabled policies.

        MPI commands are wrapped in the `ibrun` launcher, which incurs
        appreciable latency, using the format string recommended by TACC::

            ibrun

        """
    	# TODO: take the number of processors to place
        return 'ibrun'

    @property
    def always_wrap(self):
        """bool False

        TACC executes directly on compute resources via SLURM.

        """
        return False

class Stampede2(core.Environment):
    """Stampede2 environment.

    This environment supports Stampede2 at TACC (`stampede2.tacc.utexas.edu`).
    `stampede2` is an Intel CPU compute cluster having both XX-core KNL nodes
    and XX-core Skylake CPU nodes. It is recommended to define policies having
    no more than XX **total** compute threads per KNL node or XX **total**
    compute threads per SKX node.

    """

    @property
    def mpiexec(self):
        """str Format string for MPI-enabled policies.

        MPI commands are wrapped in the `ibrun` launcher, which incurs
        appreciable latency, using the format string recommended by TACC::

            ibrun

        """
    	# TODO: take the number of processors to place
        return 'ibrun'

    @property
    def always_wrap(self):
        """bool False

        TACC executes directly on compute resources via SLURM.

        """
        return False
