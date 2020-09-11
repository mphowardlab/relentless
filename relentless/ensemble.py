__all__ = ['Ensemble','RDF',
           'Volume',
           'Cube','Cuboid','Parallelepiped','TriclinicBox'
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

    @abc.abstractmethod
    def to_json(self):
        """Abstract method to serialize a :py:class:`Volume` object into a JSON equivalent."""
        pass

    @classmethod
    @abc.abstractmethod
    def from_json(cls, data):
        """Abstract method to serialize JSON data into a :py:class:`Volume` object."""
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

    def to_json(self):
        """Serializes :py:class:`Parallelepiped` object into JSON dictionary equivalent.

        Returns
        -------
        dict
            The serialized version of `self`.

        """
        return {'a':self.a,
                'b':self.b,
                'c':self.c
               }

    @classmethod
    def from_json(cls, data):
        """De-serializes JSON dictionary into a new :py:class:`Parallelepiped` object.

        Parameters
        ----------
        data : dict
            The JSON-serialized equivalent of the Parallelepiped object.

        Returns
        -------
        :py:class:`Parallelepiped`
            A new Parallelepiped object constructed from the data.

        """
        return Parallelepiped(a=data['a'],b=data['b'],c=data['c'])

class TriclinicBox(Parallelepiped):
    """Triclinic box.

    A TriclinicBox is a special type of :py:class:`Parallelepiped`. The box is
    formed by creating an orthorhombic box from three vectors of length *Lx*, *Ly*, *Lz*,
    then tilting the vectors by factors `xy`, `xy`, and `xz`. The tilt factors
    result in different basis vectors depending on the convention selected from
    :py:class:`TriclinicBox.Convention`.

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
        If the convention is not `TriclinicBox.Convention.LAMMPS` or
        `TriclinicBox.Convention.HOOMD`.

    """

    class Convention(Enum):
        """Convention by which the tilt factors are applied to the basis vectors.

        Calculation of the basis vectors by the `LAMMPS convention <https://lammps.sandia.gov/doc/Howto_triclinic.html>`_:

        .. math::

            \mathbf{a} = (L_x,0,0)
            \mathbf{b} = (xy,L_y,0)
            \mathbf{c} = (xz,yz,L_z)

        Calculation of the basis vectors by the `HOOMD convention <https://hoomd-blue.readthedocs.io/en/stable/box.html>`_:

        .. math::

            \mathbf{a} = (L_x,0,0)
            \mathbf{b} = (xy*L_y,L_y,0)
            \mathbf{c} = (xz*L_z,yz*L_z,L_z)

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
        self._convention = convention
        if self._convention is TriclinicBox.Convention.LAMMPS:
            a = (Lx,0,0)
            b = (xy,Ly,0)
            c = (xz,yz,Lz)
        elif self._convention is TriclinicBox.Convention.HOOMD:
            a = (Lx,0,0)
            b = (xy*Ly,Ly,0)
            c = (xz*Lz,yz*Lz,Lz)
        else:
            raise ValueError('Triclinic convention must be TriclinicBox.Convention.LAMMPS or TriclinicBox.Convention.HOOMD')
        super().__init__(a,b,c)

    def to_json(self):
        """Serializes :py:class:`TriclinicBox` object into JSON dictionary equivalent.

        Returns
        -------
        dict
            The serialized version of `self`.

        """
        if self._convention is TriclinicBox.Convention.LAMMPS:
            xy = self.b[0]
            xz = self.c[0]
            yz = self.c[1]
        elif self._convention is TriclinicBox.Convention.HOOMD:
            xy = self.b[0]/self.b[1]
            xz = self.c[0]/self.c[2]
            yz = self.c[1]/self.c[2]
        return {'Lx':self.a[0],
                'Ly':self.b[1],
                'Lz':self.c[2],
                'xy':xy,
                'xz':xz,
                'yz':yz,
                'convention':self._convention.name
               }

    @classmethod
    def from_json(cls, data):
        """De-serializes JSON dictionary into a new :py:class:`TriclinicBox` object.

        Parameters
        ----------
        data : dict
            The JSON-serialized equivalent of the TriclinicBox object.

        Returns
        -------
        :py:class:`TriclinicBox`
            A new TriclinicBox object constructed from the data.

        Raises
        ------
        ValueError
            If the convention specified is not LAMMPS or HOOMD.

        """
        if data['convention']=='LAMMPS':
            conv = TriclinicBox.Convention.LAMMPS
        elif data['convention']=='HOOMD':
            conv = TriclinicBox.Convention.HOOMD
        else:
            return ValueError('Only LAMMPS and HOOMD conventions are supported.')
        return TriclinicBox(Lx=data['Lx'],Ly=data['Ly'],Lz=data['Lz'],
                            xy=data['xy'],xz=data['xz'],yz=data['yz'],
                            convention=conv)

class Cuboid(TriclinicBox):
    """Orthorhombic box.

    A cuboid is a special type of :py:class:`TriclinicBox`. The three box vectors
    point along the *x*, *y*, and *z* axes, so they are all orthogonal (i.e. the
    tilt factors `xy`, `xz`, and `yz` are all 0). Each vector can have a different
    length, *Lx*, *Ly*, and *Lz*.

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
        super().__init__(Lx,Ly,Lz,0,0,0)

    def to_json(self):
        """Serializes :py:class:`Cuboid` object into JSON dictionary equivalent.

        Returns
        -------
        dict
            The serialized version of `self`.

        """
        return {'Lx':self.a[0],
                'Ly':self.b[1],
                'Lz':self.c[2],
               }

    @classmethod
    def from_json(cls, data):
        """De-serializes JSON dictionary into a new :py:class:`Cuboid` object.

        Parameters
        ----------
        data : dict
            The JSON-serialized equivalent of the Cuboid object.

        Returns
        -------
        :py:class:`Cuboid`
            A new Cuboid object constructed from the data.

        """
        return Cuboid(Lx=data['Lx'],Ly=data['Ly'],Lz=data['Lz'])

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

    def to_json(self):
        """Serializes :py:class:`Cube` object into JSON dictionary equivalent.

        Returns
        -------
        dict
            The serialized version of `self`.

        """
        return {'L':self.a[0]
               }

    @classmethod
    def from_json(cls, data):
        """De-serializes JSON dictionary into a new :py:class:`Cube` object.

        Parameters
        ----------
        data : dict
            The JSON-serialized equivalent of the Cube object.

        Returns
        -------
        :py:class:`Cube`
            A new Cube object constructed from the data.

        """
        return Cube(L=data['L'])

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
        The chemical potential for each specified type (defaults to `None`).
    N : `dict`
        The number of particles for each specified type (defaults to `None`).
    kB : float
        Boltzmann constant (defaults to 1.0).

    Raises
    ------
    ValueError
        If neither `P` nor `V` is set.
    ValueError
        If both `P` and `V` are set.
    TypeError
        If all values of `N` are not integers or `None`.
    ValueError
        If both `mu` and `N` are set for a type.

    """
    def __init__(self, T, P=None, V=None, mu=None, N=None, kB=1.0):
        if mu is None:
            mu = {}
        if N is None:
            N = {}

        types = list(mu.keys()) + list(N.keys())
        types = tuple(set(types))
        self.rdf = core.PairMatrix(types=types)

        # type checking
        if P is None and V is None:
            raise ValueError('Either P or V must be set.')
        elif P is not None and V is not None:
            raise ValueError('Both P and V cannot be set.')

        if not all([(isinstance(n,int) or n is None) for n in N.values()]):
            raise TypeError('All values of N must be integers or None.')
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
        """:py:class:`Volume`: The volume of the system."""
        return self._V

    @V.setter
    def V(self, value):
        if value is not None and not isinstance(value, Volume):
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

        No changes should be made to this dictionary, as the constant properties
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
        """Saves the parameters of the ensemble (`self`) and its RDF values into a JSON file.

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
                'mu':self.mu.todict(),
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
        with open('{}.json'.format(filename),'w') as f:
            json.dump(data, f, indent=4)

    @classmethod
    def from_file(cls, filename):
        """Constructs a new :py:class:`Ensemble` object from the ensemble parameters
        and RDF values in a JSON file.

        Parameters
        ----------
        filename : str
            The name of the file from which to load data.

        Returns
        -------
        :py:class:`Ensemble`
            An ensemble object with parameter values from the JSON files.

        """
        with open('{}.json'.format(filename)) as f:
            data = json.load(f)

        # retrieve thermodynamic parameters
        thermo = {'T':data['T'],
                  'P':data['P'],
                  'V':globals()[data['V']['__name__']].from_json(data['V']['data']),
                  'mu':data['mu'],
                  'N':data['N'],
                  'kB':data['kB']
                 }

        # create Ensemble with constant variables only
        thermo_const = copy.deepcopy(thermo)
        for var in data['constant']:
            if var=='mu' or var=='N':
                for t in data['constant'][var]:
                    if not data['constant'][var][t]:
                        thermo_const[var][t] = None
            else:
                if not data['constant'][var]:
                    thermo_const[var] = None
        ens = Ensemble(T=thermo_const['T'],
                       P=thermo_const['P'],
                       V=thermo_const['V'],
                       mu=thermo_const['mu'],
                       N=thermo_const['N'],
                       kB=thermo_const['kB'])

        # set values of conjugate variables
        for var in thermo:
            if var=='mu' or var=='N':
                for t in thermo[var]:
                    getattr(ens,var).update({t:thermo[var][t]})
            else:
                setattr(ens, var, thermo[var])

        # set rdf values
        for pair in ens.rdf:
            if data['rdf'][str(pair)] is None:
                continue
            r = [i[0] for i in data['rdf'][str(pair)]]
            g = [i[1] for i in data['rdf'][str(pair)]]
            ens.rdf[pair] = RDF(r=r, g=g)

        return ens
