"""
Optimization
============

.. toctree::
    :maxdepth: 1

    objective
    method

"""
from .method import (Optimizer,
                     SteepestDescent,
                     LineSearch)

from .objective import (ObjectiveFunction)
