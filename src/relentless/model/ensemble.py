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

from relentless import math, mpi
from relentless.collections import FixedKeyDict, PairMatrix

from . import extent


class RDF:
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
        self._spline = math.AkimaSpline(r, g)

    def __call__(self, r):
        return self._spline(r)

    @property
    def table(self):
        return self._spline.table


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
            if not isinstance(t, str):
                raise TypeError("Particle type must be a string")
        self._N = FixedKeyDict(keys=types)
        self.N.update(N)

        # P-V
        self.P = P
        self.V = V

        # rdf per-pair
        self._rdf = PairMatrix(types)

    @classmethod
    def from_json(cls, data):
        """Create from JSON data.

        It is assumed that the data is compatible with the object.

        Parameters
        ----------
        data : dict
            JSON data for ensemble.

        """
        # create initial ensemble
        thermo = {
            "T": data["T"],
            "N": data["N"],
            "P": data["P"],
        }
        if data["V"] is not None:
            ExtentType = getattr(extent, data["V"]["__name__"])
            thermo["V"] = ExtentType.from_json(data["V"]["data"])
        ens = Ensemble(**thermo)

        # unpack rdfs
        for pair in ens.rdf:
            pair_ = str(pair)
            if data["rdf"][pair_] is None:
                continue
            r = [i[0] for i in data["rdf"][pair_]]
            g = [i[1] for i in data["rdf"][pair_]]
            ens.rdf[pair] = RDF(r, g)

        return ens

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
        return cls.from_json(data)

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
            raise TypeError("V can only be set as an Extent object or as None.")
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
        r""":class:`~relentless.collections.FixedKeyDict`:
        Number of particles of each type.
        """
        return self._N

    @property
    def types(self):
        r"""tuple: The types in the ensemble."""
        return tuple(self.N.keys())

    @property
    def rdf(self):
        r""":class:`~relentless.collections.PairMatrix`:
        Radial distribution function per pair.
        """
        return self._rdf

    def copy(self):
        r"""Make a copy of the ensemble.

        Returns
        -------
        :class:`Ensemble`
            A new Ensemble object having the same state.

        """
        return copy.deepcopy(self)

    def to_json(self):
        """Export to a JSON-compatible dictionary.

        Returns
        -------
        dict
            Ensemble.

        """
        data = {
            "T": float(self.T) if self.T is not None else None,
            "N": {i: int(Ni) if Ni is not None else None for i, Ni in self.N.items()},
            "P": float(self.P) if self.P is not None else None,
            "rdf": {},
        }
        if self.V is not None:
            data["V"] = {"__name__": type(self.V).__name__, "data": self.V.to_json()}
        else:
            data["V"] = None

        # set the rdf values in data
        for pair in self.rdf:
            if isinstance(self.rdf[pair], RDF):
                data["rdf"][str(pair)] = self.rdf[pair].table.tolist()
            else:
                data["rdf"][str(pair)] = None

        return data

    def save(self, filename):
        r"""Save as a JSON file.

        Parameters
        ----------
        filename : str
            The name of the file to save data in.

        """
        data = self.to_json()
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
