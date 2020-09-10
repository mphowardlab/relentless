__all__ = ['Ensemble','RDF',
           'Volume',
           'Cube','Cuboid','Parallelepiped','Triclinic'
          ]

import abc
import copy
from enum import Enum
import json

import numpy as np

from . import core

class RDF(core.Interpolator):
    r"""Radial distribution function.

    Represents the pair distribution function :math:`g(r)` as an `nx2` table that
    can also be smoothly interpolated.

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

class Volume(abc.ABC):
    """Abstract base class defining a region of space."""
    @property
    @abc.abstractmethod
    def volume(self):
        """Volume of the region."""
        pass

class Parallelepiped(Volume):
    r"""Parallelepiped box defined by three vectors.

    The three vectors **a**, **b**, and **c** must form a right-hand basis
    so that its volume is positive:

    .. math::

        V = (\mathbf{a} \cross \mathbf{b}) \cdot \mathbf{c} > 0

    Parameters
    ----------
    a : array_like
        First vector defining the parallelepiped.
    b : array_like
        Second vector defining the parallelepiped.
    c : array_like
        Third vector defining the parallelepiped.

    Raises
    ------
    TypeError
        If a, b, and c are not all 3-element vectors.
    ValueError
        If the volume is not positive.

    """
    def __init__(self, a, b, c):
        self.a = np.asarray(a,dtype=np.float64)
        self.b = np.asarray(b,dtype=np.float64)
        self.c = np.asarray(c,dtype=np.float64)
        if not (self.a.shape==(3,) and self.b.shape==(3,) and self.c.shape==(3,)):
            raise TypeError('a, b, and c must be 3-element vectors.')
        if self.volume <= 0:
            raise ValueError('The volume must be positive.')

    @property
    def volume(self):
        """float: Volume computed using scalar triple product."""
        return np.dot(np.cross(self.a,self.b),self.c)

class Triclinic(Parallelepiped):
    """Triclinic box.

    A triclinc box is a special type of :py:class:`Parallelepiped`. The triclinic box
    is formed by creating an orthorhombic box (with the geometry of a :py:class:`Cuboid`)
    from three vectors of length *Lx*, *Ly*, *Lz*, then tilting the vectors by factors
    `xy`, `xy`, and `xz`. The tilt factors result in different basis vectors depending
    on the convention selected from :py:class:`Triclinic.Convention`.

    Parameters
    ----------
    Lx : float
        Length along the *x* axis.
    Ly : float
        Length along the *y* axis.
    Lz : float
        Length along the *z* axis.
    xy : float
        First tilt factor.
    xz : float
        Second tilt factor.
    yz : float
        Third tilt factor.

    Raises
    ------
    ValueError
        If *Lx*, *Ly*, *Lz* are not all positive.
    ValueError
        If the convention is not `Triclinic.Convention.LAMMPS` or
        `Triclinic.Convention.HOOMD`.

    """

    class Convention(Enum):
        """Convention by which the tilt factors are applied to the basis factors.

        Calculation of the basis vectors by the LAMMPS convention:

        .. math::

            `a = (Lx,0,0)`
            `b = (xy,Ly,0)`
            `c = (xz,yz,Lz)`

        Calculation of the basis vectors by the HOOMD convention:

        .. math::

            `a = (Lx,0,0)`
            `b = (xy*Ly,Ly,0)`
            `c = (xz*Lz,yz*Lz,Lz)`

        Attributes
        ----------
        LAMMPS : int
            LAMMPS convention for applying the tilt factors.
        HOOMD : int
            HOOMD convention for applying the tilt factors.

        """
        LAMMPS = 1
        HOOMD = 2

    def __init__(self, Lx, Ly, Lz, xy, xz, yz, convention=Convention.LAMMPS):
        if Lx<=0 or Ly<=0 or Lz<= 0:
            raise ValueError('All side lengths must be positive.')
        if convention is Triclinic.Convention.LAMMPS:
            a = (Lx,0,0)
            b = (xy,Ly,0)
            c = (xz,yz,Lz)
        elif convention is Triclinic.Convention.HOOMD:
            a = (Lx,0,0)
            b = (xy*Ly,Ly,0)
            c = (xz*Lz,yz*Lz,Lz)
        else:
            raise ValueError('Triclinic convention must be Triclinic.Convention.LAMMPS or Triclinic.Convention.HOOMD')
        super().__init__(a,b,c)

class Cuboid(Parallelepiped):
    """Orthorhombic box.

    A cuboid is a special type of :py:class:`Parallelepiped`. The three box vectors
    point along the *x*, *y*, and *z* axes, so they are all orthogonal. Each vector
    can have a different length, *Lx*, *Ly*, and *Lz*.

    Parameters
    ----------
    Lx : float
        Length along the *x* axis.
    Ly : float
        Length along the *y* axis.
    Lz : float
        Length along the *z* axis.

    Raises
    ------
    ValueError
        If *Lx*, *Ly*, *Lz* are not all positive.

    """
    def __init__(self, Lx, Ly, Lz):
        if Lx<=0 or Ly<=0 or Lz<= 0:
            raise ValueError('All side lengths must be positive.')
        super().__init__([Lx,0,0],[0,Ly,0],[0,0,Lz])

class Cube(Cuboid):
    """Cubic box.

    A Cube is a special type of :py:class:`Cuboid` where all vectors have the
    same length *L*.

    Parameters
    ----------
    L : float
        The edge length of the cube.

    """
    def __init__(self, L):
        super().__init__(L,L,L)

class Ensemble(object):
    """Thermodynamic ensemble.

    The parameters of the ensemble are defined as follows:
        - The temperature (`T`) is always a constant.
        - Either the pressure (`P`) or the volume (`V`) can be specified.
        - Either the chemical potential (`mu`) or the particle number (`N`)
          can be specified for each type. The particle types in the Ensemble are
          determined from the keys in the `mu` or `N` dictionaries.

    The variables that are "constants" of the ensemble (e.g. *N*, *V*, and *T*
    for the canonical ensemble) are determined from the arguments specified in the
    constructor. The fluctuating (conjugate) variables bmust be set subsequently.
    It is an error to set both variables in a conjugate pair (e.g. *P* and *V)
    during construction.

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
    ValueError
        If both `P` and `V` are set.
    TypeError
        If all values of `N` are not integers.
    ValueError
        If both `mu` and `N` are set for a type.

    """
    def __init__(self, T, P=None, V=None, mu={}, N={}, kB=1.0):
        types = list(mu.keys()) + list(N.keys())
        types = tuple(set(types))
        self.rdf = core.PairMatrix(types=types)

        # type checking
        if V is not None and not isinstance(V, Volume):
            raise TypeError('V can only be set as a Volume object.')
        elif P is None and V is None:
            raise ValueError('Either P or V must be set.')
        elif P is not None and V is not None:
            raise ValueError('Both P and V cannot be set.')
        if not all([isinstance(n, int) for n in N.values()]):
            raise TypeError('All values of N must be integers.')
        for t in types:
            if mu.get(t) is not None and N.get(t) is not None:
                raise ValueError('Both mu and N cannot be set for type {}.'.format(t))

        # temperature
        self.kB = kB
        self.T = T

        # P-V
        self.P = P
        self.V = V

        # mu-N
        self._mu = core.FixedKeyDict(keys=types)
        self._N = core.FixedKeyDict(keys=types)
        self.mu.update(mu)
        self.N.update(N)

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
        if not (value is None or isinstance(value, Volume)):
            raise TypeError('V can only be set as a Volume object or as None.')
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
        """:py:class:`FixedKeyDict`: Number of particles of each type."""
        return self._N

    @property
    def mu(self):
        """:py:class:`FixedKeyDict`: Chemical potential for each type."""
        return self._mu

    @property
    def types(self):
        """tuple: The types in the ensemble."""
        return self.N.keys

    @property
    def constant(self):
        """dict: The constant variables in the system.

        No changse should be made to this dictionary, as the constant properties
        are fixed at construction.

        """
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
        for pair in self.rdf:
            self.rdf[pair] = None

        return self

    def copy(self):
        """Makes a copy of the ensemble.

        Returns
        -------
        :py:class:`Ensemble`
            A copy of the ensemble object.

        """
        return copy.deepcopy(self)

    def save(self, filename):
        """Saves the ensemble parameters and the RDF values into a JSON file.

        Parameters
        ----------
        filename : `str`
            The name of the file to save data in.

        """
        data = []

        # append thermo data
        thermo = {'T': self.T,
                  'P': self.P,
                  'V': self.V,
                  'mu':self.mu,
                  'N': self.N,
                  'kB': self.kB,
                  'constant': self.constant} #need constant?
        data.append(thermo)

        # append rdf data
        rdf = {pair:self.rdf[pair].table for pair in self.rdf}
        data.append(rdf)

        # dump data to json file
        with open('{}.json'.format(filename),'w') as f:
            json.dump(data, f, indent=4)

    @classmethod
    def load(self, filename):
        """Loads ensemble parameters and RDF values from a JSON file into a new
        :py:class:`Ensemble` object.

        Parameters
        ----------
        filename : `str`
            The name of the file from which to load data.

        Returns
        -------
        :py:class:`Ensemble`
            An ensemble object with parameter values from the JSON files.

        """
        with open('{}.json'.format(filename)) as f:
            data = json.load(f)

        thermo = data[0]
        rdf = data[1]

        #reset fluctuating variables?

        ens = Ensemble(T=thermo['T'],
                       P=thermo['P'],
                       V=thermo['V'],
                       mu=thermo['mu'],
                       N=thermo['N'],
                       kB=thermo['kB'])

        for pair in ens.rdf:
            r = rdf[pair].table[:,1]
            g = rdf[pair].table[:,2]
            ens.rdf[pair] = RDF(r=r, g=g)

        return ens
