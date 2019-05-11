from . import core

class Ensemble(object):
    def __init__(self, types, kB=1.0):
        self.types = tuple(types)

        # temperature
        self.kB = kB
        self.T = None

        # P-V
        self.P = None
        self.V = None

        # mu-N
        self.mu = {}
        self.N = {}
        for i in self.types:
            self.mu[i] = None
            self.N[i] = None

        self.rdf = core.CoefficientMatrix(self.types)

    @property
    def beta(self):
        if self.T is None:
            raise ValueError('Temperature is not set.')
        return 1./(self.kB*self.T)

    def reset(self):
        raise NotImplementedError()

    def copy(self):
        raise NotImplementedError()

class NVT(Ensemble):
    def __init__(self, N, V, T, kB=1.0):
        super(NVT, self).__init__(N.keys(), kB)
        self.T = T
        self.V = V
        for i in self.types:
            self.N[i] = N[i]

    def reset(self):
        self.P = None
        for i in self.types:
            mu[i] = None
        self.rdf = core.CoefficientMatrix(self.types)

    def copy(self):
        return NVT(self.N, self.V, self.T, self.kB)

class NPT(Ensemble):
    def __init__(self, N, P, T, kB=1.0):
        super(NPT, self).__init__(N.keys(), kB)
        self.T = T
        self.P = P
        for i in self.types:
            self.N[i] = N[i]

    def reset(self):
        self.V = None
        for i in self.types:
            mu[i] = None
        self.rdf = core.CoefficientMatrix(self.types)

    def copy(self):
        return NPT(self.N, self.P, self.T, self.kB)

class muVT(Ensemble):
    def __init__(self, mu, V, T, kB=1.0):
        super(muVT, self).__init__(mu.keys(), kB)
        self.T = T
        self.V = V
        for i in self.types:
            self.mu[i] = mu[i]

    def reset(self):
        self.P = None
        for i in self.types:
            N[i] = None
        self.rdf = core.CoefficientMatrix(self.types)

    def copy(self):
        return muVT(self.mu, self.V, self.T, self.kB)
