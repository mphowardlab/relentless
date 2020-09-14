__all__ = ['Ensemble','RDF']

import copy
import json

import numpy as np

from .collections import FixedKeyDict,PairMatrix
from .math import Interpolator
from .volume import *

class RDF(Interpolator):
    r"""Radial distribution function.

    Represents the pair distribution function :math:`g(r)` as an ``(N,2)``
    table that can also be smoothly interpolated.

    Parameters
    ----------
    r : array_like
        1-d array of r values (continually increasing).
    g : array_like
        1-d array of g values.

    """
    def __init__(self, r, g):
        super().__init__(r,g)
        self.table = np.column_stack((r,g))

class Ensemble(object):
    r"""Thermodynamic ensemble.

    The parameters of the ensemble are defined as follows:
        - The temperature (``T``) is always a constant.
        - Either the pressure (``P``) or the volume (``V``) can be specified.
        - Either the chemical potential (``mu``) or the particle number (``N``)
          can be specified for each type. The particle types in the Ensemble are
          determined from the keys in the ``mu`` and ``N`` dictionaries.

    The variables that are "constants" of the ensemble (e.g. *N*, *V*, and *T*
    for the canonical ensemble) are determined from the arguments specified in the
    constructor. The fluctuating (conjugate) variables must be set subsequently.
    It is an error to set both variables in a conjugate pair (e.g. *P* and *V*)
    during construction.

    Parameters
    ----------
    T : float
        Temperature of the system.
    P : float
        Pressure of the system (defaults to None).
    V : :py:class:`Volume`
        Volume of the system (defaults to None).
    mu : dict
        The chemical potential for each specified type (defaults to None).
    N : dict
        The number of particles for each specified type (defaults to None).
    kB : float
        Boltzmann constant (defaults to 1.0).

    Raises
    ------
    ValueError
        If neither ``P`` nor ``V`` is set.
    ValueError
        If both ``P`` and ``V`` are set.
    TypeError
        If all values of ``N`` are not integers or None.
    ValueError
        If both ``mu`` and ``N`` are set for a type.
    ValueError
        If both ``mu`` and ``N`` are None.

    """
    def __init__(self, T, P=None, V=None, mu=None, N=None, kB=1.0):
        # P-V checking
        if P is None and V is None:
            raise ValueError('Either P or V must be set.')
        elif P is not None and V is not None:
            raise ValueError('Both P and V cannot be set.')

        # mu-N checking
        if mu is None:
            mu = dict()
        if N is None:
            N = dict()
        # type list
        types = list(mu.keys()) + list(N.keys())
        types = tuple(set(types))
        if len(types) == 0:
            raise ValueError('At least one type must be specified by N or mu.')
        for t in types:
            mu_i = mu.get(t)
            N_i = N.get(t)
            if mu_i is not None and N_i is not None:
                raise ValueError('Both mu and N cannot be set for type {}.'.format(t))
            elif N_i is not None and not isinstance(N_i,int):
                raise TypeError('Value of N for type {} must be an integer or None.'.format(t))

        # temperature
        self.kB = kB
        self.T = T

        # P-V
        self.P = P
        self.V = V

        # mu-N
        self._mu = FixedKeyDict(keys=types)
        self._N = FixedKeyDict(keys=types)
        self.mu.update(mu)
        self.N.update(N)

        # rdf per-pair
        self._rdf = PairMatrix(types=types)

        # build the set of constant variables from the constructor
        self._constant = {'T': True,
                          'P': self.P is not None,
                          'V': self.V is not None,
                          'mu': {t: (self.mu[t] is not None) for t in types},
                          'N': {t: (self.N[t] is not None) for t in types}
                         }

    @property
    def T(self):
        r"""float: The temperature."""
        return self._T

    @T.setter
    def T(self, value):
        self._T = value

    @property
    def beta(self):
        r"""float: The inverse temperature/thermal energy."""
        return 1./(self.kB*self.T)

    @property
    def V(self):
        r""":py:class:`Volume`: The volume of the system."""
        return self._V

    @V.setter
    def V(self, value):
        if value is not None and not isinstance(value, Volume):
            raise TypeError('V can only be set as a Volume object or as None.')
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
        r""":py:class:`FixedKeyDict`: Number of particles of each type."""
        return self._N

    @property
    def mu(self):
        r""":py:class:`FixedKeyDict`: Chemical potential of each type."""
        return self._mu

    @property
    def types(self):
        r"""tuple: The types in the ensemble."""
        return self.N.keys

    @property
    def constant(self):
        r"""dict: The constant variables in the system.

        Each key of the dictionary indicates which variables were defined as
        constants when the Ensemble was constructed. The user **must not**
        mutate the values in this dictionary.

        """
        return self._constant

    @property
    def rdf(self):
        r""":py:class:`PairMatrix`: Radial distribution function per pair."""
        return self._rdf

    def clear(self):
        r"""Clear the value of all conjugate (fluctuating) variables in the ensemble.

        The values of the non-constant variables and all radial distribution
        functions become None.

        Returns
        -------
        :py:class:`Ensemble`
            This Ensemble with cleared parameters.

        """
        if not self.constant['V']:
            self._V = None
        if not self.constant['P']:
            self._P = None
        for t in self.types:
            if not self.constant['N'][t]:
                self._N[t] = None
            if not self.constant['mu'][t]:
                self._mu[t] = None
        for pair in self.rdf:
            self.rdf[pair] = None

        return self

    def copy(self):
        r"""Make a copy of the ensemble.

        Returns
        -------
        :py:class:`Ensemble`
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
                'P': self.P,
                'V': {'__name__':type(self.V).__name__,
                      'data':self.V.to_json()
                     },
                'mu': self.mu.todict(),
                'N': self.N.todict(),
                'kB': self.kB,
                'constant': self.constant,
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
        :py:class:`Ensemble`
            An new Ensemble object.

        """
        with open(filename) as f:
            data = json.load(f)

        # retrieve thermodynamic parameters
        thermo = {'T': data['T'],
                  'P': data['P'],
                  'V': globals()[data['V']['__name__']].from_json(data['V']['data']),
                  'mu': data['mu'],
                  'N': data['N']
                 }

        # create Ensemble with constant variables only
        thermo_const = copy.deepcopy(thermo)
        for var in thermo:
            if var == 'mu' or var == 'N':
                for t in thermo[var]:
                    if not data['constant'][var][t]:
                        thermo_const[var][t] = None
            elif not data['constant'][var]:
                thermo_const[var] = None
        ens = Ensemble(kB=data['kB'],**thermo_const)

        # then set values of fluctuating variables
        for var in thermo:
            if var == 'mu' or var == 'N':
                for t in thermo[var]:
                    if not data['constant'][var][t]:
                        getattr(ens,var).update({t:thermo[var][t]})
            elif not data['constant'][var]:
                setattr(ens,var,thermo[var])

        # set rdf values
        for pair in ens.rdf:
            pair_ = str(pair)
            if data['rdf'][pair_] is None:
                continue
            r = [i[0] for i in data['rdf'][pair_]]
            g = [i[1] for i in data['rdf'][pair_]]
            ens.rdf[pair] = RDF(r=r, g=g)

        return ens
