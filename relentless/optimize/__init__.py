"""
Optimization
============

.. toctree::
    :maxdepth: 1

    criteria
    method
    objective

"""
from .criteria import (AllTest,
                       AndTest,
                       AnyTest,
                       ConvergenceTest,
                       GradientTest,
                       LogicTest,
                       OrTest,
                       Tolerance,
                       ValueTest)

from .method import (FixedStepDescent,
                     LineSearch,
                     Optimizer,
                     SteepestDescent)

from .objective import (ObjectiveFunction)
