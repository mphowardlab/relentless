"""Project environment management.

The environment module implements core functionalities for managing a
`relentless` project workflow. An :py:class:`Environment` is required to define
both the project data workspace and the mechanism for executing commands through
the shell. In addition to these core functionality, it also includes
specialization for various computing systems, including:

* Texas Advanced Computing Center (:py:mod:`environment.tacc`)
* Generic libraries and schedulers (:py:mod:`environment.generic`)

Environments are also intended to be extensible to individual computational
environments. Refer to the class documentation for an example of how to
implement one specific to your cluster or workstation. Contributions of new
environments are always welcome!

"""
from .core import *
from . import generic
from . import tacc
