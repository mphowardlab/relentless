"""
====================================
Optimization (`relentless.optimize`)
====================================

.. currentmodule:: relentless.optimize

Objectives
==========

.. autosummary::
    :toctree: generated/

    RelativeEntropy

Methods
=======

.. autosummary::
    :toctree: generated/

    FixedStepDescent
    SteepestDescent
    LineSearch

Convergence
===========

.. autosummary::
    :toctree: generated/

    Tolerance
    AllTest
    AndTest
    AnyTest
    GradientTest
    OrTest
    ValueTest

Developer classes
=================

.. autosummary::
    :toctree: generated/

    ObjectiveFunction
    ObjectiveFunctionResult
    Optimizer
    ConvergenceTest
    LogicTest

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

from .objective import (ObjectiveFunction,
                        ObjectiveFunctionResult,
                        RelativeEntropy)
