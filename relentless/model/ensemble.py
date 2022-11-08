"""
Ensemble
========

The behavior and properties of a simulated system can be described with a statistical
mechanical ensemble. Even though the system contains an extremely large number
of particles and distribution of possible states, the ensemble can be parametrized
by just a few quantities. Then, the real thermodynamic properties of the system
can be determined using the ensemble average of the appropriate quantity. In addition,
for each pair in the ensemble, the radial distribution function (RDF) is also defined
using an Akima spline.

.. autosummary::
    :nosignatures:

    Ensemble
    RDF

.. autoclass:: Ensemble
    :members:
.. autoclass:: RDF
    :members:

"""
import copy
import json

import numpy

from relentless.collections import FixedKeyDict,PairMatrix
from relentless.math import Interpolator
from . import extent
from relentless import mpi

class RDF(Interpolator):
    r"""Radial distribution function.

    Represents the pair distribution function :math:`g(r)` as a ``(N,2)``
    table that can also be smoothly interpolated.

    Parameters
    ----------
    r : array_like
        1-d array of :math:`r` values (continually increasing).
    g : array_like
        1-d array of :math:`g(r)` values.

    """
    def __init__(self, r, g):
        super().__init__(r,g)
        self.table = numpy.column_stack((r,g))

class Ensemble:
    r"""Thermodynamic ensemble.

    An ensemble is defined by:

        - The temperature ``T``.
        - The number ``N`` for each particle type. The particle types are
          strings determined from the keys of ``N``.
        - The extent ``V`` and/or pressure ``P``.

    Parameters
    ----------
    T : float
        Temperature of the system.
    N : dict
        Number of particles for each type.
    V : :class:`Extent`, optional
        Extent of the system.
    P : float, optional
        Pressure of the system.

    """
    def __init__(self, T, N, V=None, P=None):
        # T
        self.T = T

        # N
        types = tuple(N.keys())
        for t in types:
            if not isinstance(t,str):
                raise TypeError('Particle type must be a string')
        self._N = FixedKeyDict(keys=types)
        self.N.update(N)

        # P-V
        self.P = P
        self.V = V

        # rdf per-pair
        self._rdf = PairMatrix(types=types)

    @property
    def T(self):
        r"""float: The temperature."""
        return self._T

    @T.setter
    def T(self, value):
        self._T = value

    @property
    def V(self):
        r""":class:`~relentless.extent.Extent`: The extent of the system."""
        return self._V

    @V.setter
    def V(self, value):
        if value is not None and not isinstance(value, extent.Extent):
            raise TypeError('V can only be set as an Extent object or as None.')
        self._V = value

    @property
    def P(self):
        r"""float: The pressure of the system."""
        return self._P

    @P.setter
    def P(self, value):
        self._P = value

    @property
    def N(self):
        r""":class:`~relentless.collections.FixedKeyDict`: Number of particles of each type."""
        return self._N

    @property
    def types(self):
        r"""tuple: The types in the ensemble."""
        return self.N.keys()

    @property
    def pairs(self):
        r"""tuple: The pairs in the ensemble."""
        return self.rdf.pairs

    @property
    def rdf(self):
        r""":class:`~relentless.collections.PairMatrix`: Radial distribution function per pair."""
        return self._rdf

    def copy(self):
        r"""Make a copy of the ensemble.

        Returns
        -------
        :class:`Ensemble`
            A new Ensemble object having the same state.

        """
        return copy.deepcopy(self)

    def save(self, filename):
        r"""Save the Ensemble as a JSON file.

        Parameters
        ----------
        filename : str
            The name of the file to save data in.

        """
        data = {'T': self.T,
                'N': dict(self.N),
                'V': {'__name__':type(self.V).__name__,
                      'data':self.V.to_json()
                     },
                'P': self.P,
                'rdf': {}
               }

        # set the rdf values in data
        for pair in self.rdf:
            if self.rdf[pair]:
                data['rdf'][str(pair)] = self.rdf[pair].table.tolist()
            else:
                data['rdf'][str(pair)] = None

        # dump data to json file
        with open(filename,'w') as f:
            json.dump(data, f, indent=4)

    @classmethod
    def from_file(cls, filename):
        r"""Construct an Ensemble from a JSON file.

        Parameters
        ----------
        filename : str
            The name of the file from which to load data.

        Returns
        -------
        :class:`Ensemble`
            An new Ensemble object.

        """
        data = mpi.world.load_json(filename)

        # create initial ensemble
        ExtentType = getattr(extent,data['V']['__name__'])
        thermo = {'T': data['T'],
                  'N': data['N'],
                  'V': ExtentType.from_json(data['V']['data']),
                  'P': data['P'],
                 }
        ens = Ensemble(**thermo)

        # unpack rdfs
        for pair in ens.rdf:
            pair_ = str(pair)
            if data['rdf'][pair_] is None:
                continue
            r = [i[0] for i in data['rdf'][pair_]]
            g = [i[1] for i in data['rdf'][pair_]]
            ens.rdf[pair] = RDF(r,g)

        return ens
