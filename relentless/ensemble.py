import copy
import json

import numpy as np

from . import core

class RDF(core.Interpolator):
    def __init__(self, r, g):
        super().__init__(r,g)
        self.table = np.column_stack((r,g))

class Ensemble(object):
    def __init__(self, types, T, P=None, V=None, mu={}, N={}, kB=1.0, conjugates=None):
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

        # mu-N, must be set by type
        self._mu = core.TypeDict(self.types)
        self._N = core.TypeDict(self.types)
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

    def save(self, basename='ensemble'):
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
