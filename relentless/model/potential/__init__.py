"""
Potentials
==========

Potentials define the interactions between particles in a simulation. Currently,
the following types of potentials are supported:

.. toctree::
    :maxdepth: 1

    pair

.. automodule:: relentless.potential.potential

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
