import numpy as np

from .environment import Policy
from .ensemble import Ensemble

class RDF(object):
    def __init__(self):
        pass

    def run(self, env, step):
        raise NotImplementedError()

class Mock(RDF):
    def __init__(self, ensemble, dr, rcut, potential):
        super().__init__()
        self._ensemble = ensemble
        self.dr = dr
        self.rcut = rcut
        self.potential = potential

    def run(self, env, step):
        # r with requested spacing (accounting for possible last fractional bin)
        nbins = np.ceil(self.rcut/self.dr).astype(int)
        r = self.dr*np.arange(nbins+1)
        r[-1] = self.rcut

        # center r on the bins
        r = 0.5*(r[:-1] + r[1:])

        # use dilute g(r) approximation
        ens = self._ensemble.copy()
        for pair in ens.rdf:
            u = self.potential(r, pair)
            gr = np.exp(-ens.beta*u)
            ens.rdf[pair] = np.column_stack((r,gr))

        return ens

class LAMMPS(object):
    def __init__(self, ensemble, order, rdf='rdf.lammps', thermo='log.lammps'):
        self._ensemble = ensemble.copy()
        self._ensemble.reset()

        self.order = order
        self.rdf = rdf
        self.thermo = thermo

    def run(self, env, step):
        # default empty ensemble
        ens = self._ensemble.copy()
        ens.T = 0.
        ens.P = 0.
        ens.V = 0.
        for t in ens.types:
            ens.N[t] = 0

        with env.data(step):
            # thermodynamic properties
            with open(self.thermo) as f:
                # advance to the table section
                line = f.readline()
                while line and 'RELENTLESS PRODUCTION' not in line:
                    line = f.readline()
                while line and 'Step' not in line:
                    line = f.readline()

                header = line.strip().split()
                n_entry = len(header)
                num_samples = 0
                line = f.readline()
                while line and 'Loop time' not in line:
                    row = line.strip().split()
                    if len(row) != n_entry:
                        raise IOError('Read bad row of thermo data')

                    ens.T += float(row[self.order['T']])
                    ens.P += float(row[self.order['P']])
                    ens.V += float(row[self.order['V']])
                    for t in ens.types:
                        ens.N[t] += int(row[self.order['N_'+t]])
                    num_samples += 1

                    line = f.readline()

                # normalize mean
                if num_samples > 0:
                    ens.T /= num_samples
                    ens.P /= num_samples
                    ens.V /= num_samples
                    for t in ens.types:
                        ens.N[t] /= num_samples
                else:
                    raise IOError('LAMMPS thermo table empty!')

            # radial distribution functions
            rdf = np.loadtxt(self.rdf, skiprows=4)
            rs = rdf[:,1]
            for i,pair in enumerate(ens.rdf):
                ens.rdf[pair] = np.column_stack((rs,rdf[:,2+2*i]))

        return ens
