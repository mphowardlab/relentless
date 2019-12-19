import copy
import numpy as np

from . import core

class RDF(core.Interpolator):
    def __init__(self, r, g):
        super().__init__(r,g)
        self.table = np.column_stack((r,g))

class Ensemble(object):
    def __init__(self, types, T, P=None, V=None, mu={}, N={}, kB=1.0):
        self.types = tuple(types)
        self.rdf = core.PairMatrix(self.types)

        # temperature
        self.kB = kB
        self.T = T

        # P-V
        self._P = P
        self._V = V
        if self.P is None and self.V is None:
            raise ValueError('Either P or V must be set.')
        elif self.P is not None and self.V is not None:
            raise ValueError('Both P and V cannot be set.')

        # mu-N, must be set by species
        self._mu = core.TypeDict(self.types)
        self._N = core.TypeDict(self.types)
        for t in self.types:
            mut = t in mu
            Nt = t in N

            if not mut and not Nt:
                raise ValueError('Either mu or N must be set for type {}.'.format(t))
            elif mut and Nt:
                raise ValueError('Both mu and N cannot be set for type {}.'.format(t))
            elif Nt:
                self._N[t] = N[t]
            else:
                self._mu[t] = mu[t]

        # save the set of conjugate variables from the constructor
        conjugates = []
        if self.P is None:
            conjugates.append('P')
        if self.V:
            conjugates.append('V')
        for t in self.types:
            if self.mu[t] is None:
                conjugates.append('mu_{}'.format(t))
            if self.N[t] is None:
                conjugates.append('N_{}'.format(t))
        self._conjugates = tuple(conjugates)

    @property
    def beta(self):
        return 1./(self.kB*self.T)

    @property
    def V(self):
        return self._V

    @V.setter
    def V(self, value):
        if 'V' in self.conjugates:
            self._V = value
        else:
            raise AttributeError('Volume is not a conjugate variable.')

    @property
    def P(self):
        return self._P

    @P.setter
    def P(self, value):
        if 'P' in self.conjugates:
            self._P = value
        else:
            raise AttributeError('Pressure is not a conjugate variable.')

    @property
    def N(self):
        return self._N.asdict()

    @N.setter
    def N(self, value):
        for t in value:
            if 'N_{}'.format(t) in self.conjugates:
                self._N[t] = value[t]
            else:
                raise AttributeError('Number is not a conjugate variable.')

    @property
    def mu(self):
        return self._mu.asdict()

    @mu.setter
    def mu(self, value):
        for t in self.types:
            if 'mu_{}'.format(t) in self.conjugates:
                self._mu[t] = value[t]
            else:
                raise AttributeError('Chemical potential is not a conjugate variable.')

    @property
    def conjugates(self):
        return self._conjugates

    def reset(self):
        if 'V' in self.conjugates:
            self._V = None
        if 'P' in self.conjugates:
            self._P = None
        for t in self.types:
            if 'N_{}'.format(t) in self.conjugates:
                self._N[t] = None
            if 'mu_{}'.format(t) in self.conjugates:
                self._mu[t] = None
        self.rdf = core.PairMatrix(self.types)

        return self

    def copy(self):
        ens = copy.deepcopy(self)
        return ens.reset()
