__all__ = ['Ensemble','RDF']

import copy
import json

import numpy as np

from . import core

class RDF(core.Interpolator):
    """Constructs a :py:class:`Interpolator` object interpolating through an RDF
    function :math:`g(r)`, and creates an `nx2` array with columns as `r` and `g`.

    Parameters
    ----------
    r : array_like
        1-d array of r values, that must be continually increasing.
    g : array_like
        1-d array of g values.

    """
    def __init__(self, r, g):
        super().__init__(r,g)
        self.table = np.column_stack((r,g))

class Ensemble(object):
    """Constructs a thermodynamic ensemble for a set of types with specified parameters.

    Parameters
    ----------
    types : array_like
        List of types (A type must be a `str`).
    T : float or int
        Temperature of the system.
    P : float or int
        Pressure of the system (defaults to `None`).
    V : float or int
        Volume of the system (defaults to `None`).
    mu : `dict`
        The chemical potential for each specified type (defaults to empty `dict`).
    N : `dict`
        The number of particles for each specified type (defaults to empty `dict`.
    kB : float
        Boltzmann constant (defaults to 1.0).
    conjugates : array_like
        A list of the fluctuating variables conjugate to the constant parameters
        of the ensemble (defaults to `None`).

    Raises
    ------
    ValueError
        If either `P` or `V` are not set.
    ValueError
        If either `mu` or `N` are not set for each type.
    ValueError
        If the conjugates are not set, and both `P` and `V` are set.
    ValueError
        If the conjugates are not set, and both `mu` and `N` are set for each type.

    """
    def __init__(self, types, T, P=None, V=None, mu={}, N={}, kB=1.0, conjugates=None):
        self.types = tuple(types)
        self.rdf = core.PairMatrix(types=self.types)

        # temperature
        self.kB = kB
        self.T = T

        # P-V
        self._P = P
        self._V = V
        if self.P is None and self.V is None:
            raise ValueError('Either P or V must be set.')

        # mu-N, must be set by type
        self._mu = core.FixedKeyDict(keys=self.types)
        self._N = core.FixedKeyDict(keys=self.types)
        for t in self.types:
            if (t not in mu or mu[t] is None) and (t not in N or N[t] is None):
                raise ValueError('Either mu or N must be set for type {}.'.format(t))
            if t in N:
                self._N[t] = N[t]
            if t in mu:
                self._mu[t] = mu[t]

        # conjugates can be specified (and assumed correct), or they can be deduced
        if conjugates is not None:
            self._conjugates = tuple(conjugates)
        else:
            if self.P is not None and self.V is not None:
                raise ValueError('Both P and V cannot be set.')
            for t in self.types:
                if self.mu[t] is not None and self.N[t] is not None:
                    raise ValueError('Both mu and N cannot be set for type {}.'.format(t))

            # build the set of conjugate variables from the constructor
            conjugates = []
            if self.P is None:
                conjugates.append('P')
            if self.V is None:
                conjugates.append('V')
            for t in self.types:
                if self.mu[t] is None:
                    conjugates.append('mu_{}'.format(t))
                if self.N[t] is None:
                    conjugates.append('N_{}'.format(t))
            self._conjugates = tuple(conjugates)

    @property
    def beta(self):
        """float: The inverse temperature/thermal energy."""
        return 1./(self.kB*self.T)

    @property
    def V(self):
        """float or int: The volume of the system; raises an AttributeError if
        volume is not a conjugate variable."""
        return self._V

    @V.setter
    def V(self, value):
        if 'V' in self.conjugates:
            self._V = value
        else:
            raise AttributeError('Volume is not a conjugate variable.')

    @property
    def P(self):
        """float or int: The pressure of the system; raises an AttributeError if
        pressure is not a conjugate variable."""
        return self._P

    @P.setter
    def P(self, value):
        if 'P' in self.conjugates:
            self._P = value
        else:
            raise AttributeError('Pressure is not a conjugate variable.')

    @property
    def N(self):
        """dict: The number of particles for each specified type."""
        return self._N.todict()

    @N.setter
    def N(self, value):
        for t in value:
            if 'N_{}'.format(t) in self.conjugates:
                self._N[t] = value[t]
            else:
                raise AttributeError('Number is not a conjugate variable.')

    @property
    def mu(self):
        """dict: The chemical potential for each specified type."""
        return self._mu.todict()

    @mu.setter
    def mu(self, value):
        for t in value:
            if 'mu_{}'.format(t) in self.conjugates:
                self._mu[t] = value[t]
            else:
                raise AttributeError('Chemical potential is not a conjugate variable.')

    @property
    def conjugates(self):
        """array_like: A list of the fluctuating variables conjugate to the
        constant parameters of the ensemble."""
        return self._conjugates

    def reset(self):
        """Resets all conjugate variables in the ensemble and the *rdf parameter* to `None`.

        Returns
        -------
        :py:class:`Ensemble`
            Returns the ensemble object with reset parameters.

        """
        if 'V' in self.conjugates:
            self._V = None
        if 'P' in self.conjugates:
            self._P = None
        for t in self.types:
            if 'N_{}'.format(t) in self.conjugates:
                self._N[t] = None
            if 'mu_{}'.format(t) in self.conjugates:
                self._mu[t] = None
        self.rdf = core.PairMatrix(types=self.types)

        return self

    def copy(self):
        """Copies the ensemble and returns the copy with reset parameters.

        Returns
        -------
        :py:class:`Ensemble`
            A copy of the ensemble object with reset parameters.

        """
        ens = copy.deepcopy(self)
        return ens.reset()

    def save(self, basename='ensemble'):
        """Saves the values of the ensemble parameters into a JSON file,
        and saves the RDF values into separate JSON files.

        Parameters
        ----------
        basename : `str`
            The basename of the files in which to save data (defaults to 'ensemble').

        """
        # dump thermo data to json file
        data = {'types': self.types,
                'kB': self.kB,
                'T': self.T,
                'P': self.P,
                'V': self.V,
                'N': self.N,
                'mu': self.mu,
                'conjugates': self.conjugates}
        with open('{}.json'.format(basename),'w') as f:
            json.dump(data, f, sort_keys=True, indent=4)

        # dump rdfs in separate files
        for pair in self.rdf:
            if self.rdf[pair] is not None:
                i,j = pair
                np.savetxt('{}.{}.{}.dat'.format(basename,i,j), self.rdf[pair].table, header='r g[{},{}](r)'.format(i,j))

    @classmethod
    def load(self, basename='ensemble'):
        """Loads ensemble parameter values, and RDF values from JSON files into
        a new :py:class:`Ensemble` object.

        Parameters
        ----------
        basename : `str`
            The basename of the files from which to save data (defaults to 'ensemble').

        Returns
        -------
        :py:class:`Ensemble`
            An ensemble object with parameter values from the JSON files.

        Raises
        ------
        FileNotFoundError
            If a data file is not found for a specific RDF pair value.

        """
        with open('{}.json'.format(basename)) as f:
            data = json.load(f)

        ens = Ensemble(types=data['types'],
                       T=data['T'],
                       P=data['P'],
                       V=data['V'],
                       N=data['N'],
                       mu=data['mu'],
                       kB=data['kB'],
                       conjugates=data['conjugates'])

        for pair in ens.rdf:
            try:
                gr = np.loadtxt('{}.{}.{}.dat'.format(basename,pair[0],pair[1]))
                ens.rdf[pair] = RDF(gr[:,0], gr[:,1])
            except FileNotFoundError:
                pass

        return ens
