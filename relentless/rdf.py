import numpy as np

class RDF(object):
    def __init__(self, env):
        self.env = env

    def __call__(self, trajectory, pair, dr, rcut):
        raise NotImplementedError()

class AllPairsRDF(RDF):
    def __init__(self, env):
        super(AllPairsRDF, self).__init__(env)

    def __call__(self, trajectory, pair, dr, rcut):
        # r with requested spacing (accounting for possible last fractional bin)
        nbins = np.ceil(rcut/dr).astype(int)
        r = dr*np.arange(nbins+1)
        r[-1] = rcut

        # volume of spherical shell bins
        vbins = (4.*np.pi/3.)*(r[1:]**3-r[:-1]**3)

        # center r on the bins
        r = 0.5*(r[:-1] + r[1:])
        gr = np.zeros(nbins, dtype=np.float64)

        for s in trajectory:
            if rcut > 0.5*np.min(s.box.L):
                raise ValueError('RDF cutoff exceeds half the shortest box length.')

            # select out the right particles
            type_i = s.types == pair[0]
            type_j = s.types == pair[1]

            # number of each
            N_i = np.sum(type_i)
            N_j = np.sum(type_j)
            if N_i == 0 or N_j == 0:
                continue

            tags = np.arange(s.N)
            tagsi = tags[type_i]
            tagsj = tags[type_j]

            ris = s.positions[type_i]
            rjs = s.positions[type_j]

            # iterate through all pairs
            overlap = 0
            rcut2 = rcut*rcut
            counts = np.zeros(nbins, dtype=np.int64)
            for i,ri in zip(tagsi,ris):
                for j,rj in zip(tagsj,rjs):
                    if i == j:
                        overlap += 1
                        continue

                    # distance check
                    drij = rj - ri
                    drij = s.box.min_image(drij)
                    drij2 = np.sum(drij*drij)

                    if drij2 < rcut2:
                        bin_ = int(np.sqrt(drij2)/dr)
                        assert bin_ < nbins, "Bin {} outside of allocated RDF range ({})".format(bin_,nbins)
                        counts[bin_] += 1

            gr += counts*s.box.volume/((N_i*N_j-overlap)*vbins)
        gr /= len(trajectory)

        return r,gr
