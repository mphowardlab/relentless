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

Algorithms
==========

.. autosummary::
    :toctree: generated/

    SteepestDescent
    FixedStepDescent
    LineSearch

Convergence criteria
====================

.. autosummary::
    :toctree: generated/

    Tolerance
    ValueTest
    GradientTest
    AllTest
    AndTest
    AnyTest
    OrTest

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

from .criteria import (
    AllTest,
    AndTest,
    AnyTest,
    ConvergenceTest,
    GradientTest,
    LogicTest,
    OrTest,
    Tolerance,
    ValueTest,
)
from .method import FixedStepDescent, LineSearch, Optimizer, SteepestDescent
from .objective import ObjectiveFunction, ObjectiveFunctionResult, RelativeEntropy
