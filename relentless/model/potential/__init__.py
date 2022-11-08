"""
=========================================
Potentials (`relentless.model.potential`)
=========================================

.. currentmodule:: relentless.model.potential

Pair potentials
===============

.. autosummary::
    :toctree: generated/

    Depletion
    LennardJones
    PairSpline
    Yukawa

Developer classes
=================

.. autosummary::
    :toctree: generated/

    Potential
    Parameters
    PairPotential
    PairParameters

"""
from .potential import (
    Parameters,
    Potential,
    )

from .pair import (
    Depletion,
    LennardJones,
    PairParameters,
    PairPotential,
    PairSpline,
    Yukawa,
    )
