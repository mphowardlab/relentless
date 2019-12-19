import numpy as np

from .environment import Policy
from .ensemble import Ensemble
from .ensemble import RDF

class Mock:
    def __init__(self, ensemble, dr, rcut, potential):
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
            ens.rdf[pair] = RDF(r,gr)

        return ens

class LAMMPS(object):
    def __init__(self, ensemble, order, rdf='rdf.lammps', thermo='log.lammps'):
        self._ensemble = ensemble.copy()
        self.order = order
        self.rdf = rdf
        self.thermo = thermo

    def run(self, env, step):
        with env.data(step):
            # thermodynamic property accumulators
            T = 0.
            P = 0.
            V = 0.
            N = {}
            for t in self._ensemble.types:
                N[t] = 0

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
                        raise IOError('Read bad row of LAMMPS thermo data')

                    T += float(row[self.order['T']])
                    P += float(row[self.order['P']])
                    V += float(row[self.order['V']])
                    for t in self._ensemble.types:
                        N[t] += int(row[self.order['N_'+t]])
                    num_samples += 1

                    line = f.readline()

                # normalize mean
                if num_samples > 0:
                    T /= num_samples
                    P /= num_samples
                    V /= num_samples
                    for t in N:
                        N[t] /= float(num_samples)
                else:
                    raise IOError('LAMMPS thermo table empty!')

            # set conjugate variables as needed for ensemble
            ens = self._ensemble.copy()
            if 'P' in ens.conjugates:
                ens.P = P
            if 'V' in ens.conjugates:
                ens.V = V
            if 'N' in ens.conjugates:
                ens.N = N

            # radial distribution functions
            rdf = np.loadtxt(self.rdf, skiprows=4)
            rs = rdf[:,1]
            for i,pair in enumerate(ens.rdf):
                ens.rdf[pair] = RDF(rs,rdf[:,2+2*i])

        return ens
