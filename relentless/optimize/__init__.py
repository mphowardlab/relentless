"""
Optimization
============

.. toctree::
    :maxdepth: 1

    objective
    method

"""
from .method import (Optimizer,
                     LineSearch,
                     SteepestDescent,
                     FixedStepDescent)

from .objective import (ObjectiveFunction)
