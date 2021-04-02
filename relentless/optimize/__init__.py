"""
Optimization
============

.. toctree::
    :maxdepth: 1

    objective
    method

"""
from .method import (FixedStepDescent,
                     LineSearch,
                     Optimizer,
                     SteepestDescent)

from .objective import (ObjectiveFunction)

