__all__ = ['Ensemble','RDF',
           'Volume',
           'Cube','Cuboid','Parallelepiped'
          ]

import abc
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

class Volume(abc.ABC):
    """Abstract base class for a defined volume/region."""
    @property
    @abc.abstractmethod
    def volume(self):
        """Volume of the region."""
        pass

class Parallelepiped(Volume):
    """General triclinic box.

    Parameters
    ----------
    a : array_like
        One of the vectors forming the parallelepiped; must be an array
        containing elements as (`x`,`y`,`z`).
    b : array_like
        One of the vectors forming the parallelepiped; must be an array
        containing elements as (`x`,`y`,`z`).
    c : array_like
        One of the vectors forming the parallelepiped; must be an array
        containing elements as (`x`,`y`,`z`).

    Raises
    ------
    TypeError
        If a, b, and c are not all `3x1` arrays.

    """
    def __init__(self, a, b, c):
        self.a = np.asarray(a)
        self.b = np.asarray(b)
        self.c = np.asarray(c)
        if not (self.a.shape==(3,) and self.b.shape==(3,) and self.c.shape==(3,)):
            raise TypeError('a, b, and c must be 3x1 arrays.')
        self.matrix = np.column_stack((self.a,self.b,self.c))

    @property
    def volume(self):
        """float: Volume computed using scalar triple product."""
        return np.linalg.norm(np.dot(np.cross(self.a,self.b),self.c))

class Cuboid(Parallelepiped):
    """Orthorhombic box.

    Parameters
    ----------
    Lx : float
        The length of the cuboid.
    Ly : float
        The width of the cuboid.
    Lz : float
        The height of the cuboid.

    Raises
    ------
    TypeError
        If Lx, Ly, and Lz are not all integers or floats.

    """
    def __init__(self, Lx, Ly, Lz):
        if not (isinstance(Lx,(float,int)) and isinstance(Ly,(float,int)) and isinstance(Lz,(float,int))):
            raise TypeError('Lx, Ly, and Lz must all be ints or floats.')
        super().__init__([Lx,0,0],[0,Ly,0],[0,0,Lz])

class Cube(Cuboid):
    """Cubic box.

    Parameters
    ----------
    L : float
        The edge length of the cube.

    Raises
    ------
    TypeError
        If L is not an integer or float.

    """
    def __init__(self, L):
        if not isinstance(L,(float,int)):
            raise TypeError('L must be an int or float.')
        super().__init__(L,L,L)

class Ensemble(object):
    """Thermodynamic ensemble.

    The parameters of the ensemble are defined as follows:
        - The temperature (`T`) is always a constant.
        - Either the pressure (`P`) or the volume (`V`) can be specified.
        - Either the chemical potential (`mu`) or the particle number (`N`)
          can be specified for each type. The types are only defined in the
          `mu` or `N` dictionaries.

    Parameters
    ----------
    T : float
        Temperature of the system.
    P : float
        Pressure of the system (defaults to `None`).
    V : :py:class:`Volume`
        Volume of the system (defaults to `None`).
    mu : `dict`
        The chemical potential for each specified type (defaults to empty `dict`).
    N : `dict`
        The number of particles for each specified type (defaults to empty `dict`).
    kB : float
        Boltzmann constant (defaults to 1.0).

    Raises
    ------
    TypeError
        If `V` is not set as a :py:class:`Volume` object.
    ValueError
        If neither `P` nor `V` is set.
    TypeError
        If all values of `N` are not integers.
    ValueError
        If both `P` and `V` are set.
    ValueError
        If both `mu` and `N` are set for a type.

    """
    def __init__(self, T, P=None, V=None, mu={}, N={}, kB=1.0):
        types = list(mu.keys()) + list(N.keys())
        types = tuple(set(types))
        self.rdf = core.PairMatrix(types=types)

        # temperature
        self.kB = kB
        self.T = T

        # P-V
        self._P = P
        if V is not None and not isinstance(V, Volume):
            raise TypeError('V can only be set as a Volume object.')
        self._V = V
        if self.P is None and self.V is None:
            raise ValueError('Either P or V must be set.')

        # mu-N
        self._mu = core.FixedKeyDict(keys=types)
        self._N = core.FixedKeyDict(keys=types)
        self._mu.update(mu)
        self._N.update(N)
        for n in N.values():
            if not isinstance(n, int):
                raise TypeError('All values of N must be integers.')

        # check that both parameters in a conjugate pair are not set for a type
        if self.P is not None and self.V is not None:
            raise ValueError('Both P and V cannot be set.')
        for t in types:
            if self.mu[t] is not None and self.N[t] is not None:
                raise ValueError('Both mu and N cannot be set for type {}.'.format(t))

        # build the set of constant variables from the constructor
        self._constant = {'P': self.P is not None,
                          'V': self.V is not None,
                          'mu': {t:(self.mu[t] is not None) for t in types},
                          'N': {t:(self.N[t] is not None) for t in types}
                         }

    @property
    def beta(self):
        """float: The inverse temperature/thermal energy."""
        return 1./(self.kB*self.T)

    @property
    def V(self):
        """:py:class:`Volume`: The volume of the system, can only be set as a `Volume` object."""
        return self._V

    @V.setter
    def V(self, value):
        if not isinstance(value, Volume):
            raise TypeError('V can only be set as a Volume object.')
        self._V = value

    @property
    def P(self):
        """float or int: The pressure of the system."""
        return self._P

    @P.setter
    def P(self, value):
        self._P = value

    @property
    def N(self):
        """:py:class:`FixedKeyDict`: The number of particles for each specified type."""
        return self._N

    @property
    def mu(self):
        """:py:class:`FixedKeyDict`: The chemical potential for each specified type."""
        return self._mu

    @property
    def types(self):
        """tuple: The types in the ensemble."""
        return tuple(self._N.todict().keys())

    @property
    def constant(self):
        """dict: The constant variables in the system. READ-ONLY."""
        return self._constant

    def clear(self):
        """Clears all conjugate variables and all rdf values in the ensemble
        (set to `None`).

        Returns
        -------
        :py:class:`Ensemble`
            Returns the ensemble object with cleared parameters.

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
        self.rdf = core.PairMatrix(types=self.types)

        return self

    def copy(self):
        """Returns a copy of the specified ensemble.

        Returns
        -------
        :py:class:`Ensemble`
            A copy of the ensemble object.

        """
        return copy.deepcopy(self)

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
