"""
Optimization
============

.. toctree::
    :maxdepth: 1

    criteria
    method
    objective

"""
from .criteria import (AbsoluteGradientTest,
                       AbsoluteTolerance,
                       AllTest,
                       AnyTest,
                       ConvergenceTest,
                       LogicTest,
                       RelativeGradientTest,
                       RelativeTolerance,
                       ValueTest)

from .method import (FixedStepDescent,
                     LineSearch,
                     Optimizer,
                     SteepestDescent)

from .objective import (ObjectiveFunction)
