"""
Volumes
=======
A :class:`Volume` represents a 3D region of space with a fixed, scalar volume. It corresponds
to the "box" used in simulations. The following box types have been implemented:

.. autosummary::
    :nosignatures:

    Parallelepiped
    TriclinicBox
    Cuboid
    Cube

The :class:`TriclinicBox` can be constructed using both the LAMMPS and HOOMD-blue
:class:`TriclinicBox.Convention`\s for applying tilt factors.

Examples
--------
Construct a simulation box with defined basis vectors and volume::

    v = relentless.volume.Cube(L=3)
    >>> print(v.a)
    [3.0 0.0 0.0]
    >>> print(v.b)
    [0.0 3.0 0.0]
    >>> print(v.c)
    [0.0 0.0 3.0]
    >>> print(v.volume)
    27.0

.. rubric:: Developer notes

To implement your own simulation box, create a class that derives from :class:`Volume`
and define the required methods.

.. autosummary::
    :nosignatures:

    Volume

.. autoclass:: Volume
    :members:
.. autoclass:: Parallelepiped
    :members:
.. autoclass:: TriclinicBox
    :members:
.. autoclass:: Cuboid
    :members:
.. autoclass:: Cube
    :members:

"""
import abc
from enum import Enum

import numpy

class Extent(abc.ABC):
    r"""Abstract base class defining a region of space.

    A Volume can be any region of space; typically, it is a simulation "box."
    Any deriving class must implement the volume method that computes the scalar
    volume of the region. Additionally, methods to serialize and deserialize a
    Volume must be specified so that the object can be saved to disk.

    """
    @property
    @abc.abstractmethod
    def to_json(self):
        r"""Serialize as a dictionary.

        The serialized data can be saved to file as JSON data.

        Returns
        -------
        dict
            The serialized :class:`Volume` data.

        """
        pass

    @classmethod
    @abc.abstractmethod
    def from_json(cls, data):
        r"""Deserialize from a dictionary.

        Returns
        -------
        :class:`Volume`
            The object reconstructed from the ``data``.

        """
        pass

class Volume(Extent):
    r"""Abstract base class defining a 3d region of space.

    A Volume can be any region of space; typically, it is a simulation "box."
    Any deriving class must implement the volume method that computes the scalar
    volume of the region. Additionally, methods to serialize and deserialize a
    Volume must be specified so that the object can be saved to disk.

    """
    @property
    @abc.abstractmethod
    def volume(self):
        r"""float: Volume of the region."""
        pass

    def to_json(self):
        r"""Serialize as a dictionary.

        The serialized data can be saved to file as JSON data.

        Returns
        -------
        dict
            The serialized :class:`Volume` data.

        """
        pass

    @classmethod
    def from_json(cls, data):
        r"""Deserialize from a dictionary.

        Returns
        -------
        :class:`Volume`
            The object reconstructed from the ``data``.

        """
        pass

class Area(Extent):
    r"""Abstract base class defining a 2d region of space.

    An Area can be any 2d region of space; typically, it is a simulation "box."
    Any deriving class must implement the volume method that computes the scalar
    area of the region. Additionally, methods to serialize and deserialize a
    Area must be specified so that the object can be saved to disk.

    """
    @property
    @abc.abstractmethod
    def area(self):
        r"""float: Area of the region."""
        pass

    def to_json(self):
        r"""Serialize as a dictionary.

        The serialized data can be saved to file as JSON data.

        Returns
        -------
        dict
            The serialized :class:`Volume` data.

        """
        pass

    @classmethod
    def from_json(cls, data):
        r"""Deserialize from a dictionary.

        Returns
        -------
        :class:`Volume`
            The object reconstructed from the ``data``.

        """
        pass

class Parallelepiped(Volume):
    r"""Parallelepiped box defined by three vectors.

    The three vectors :math:`\mathbf{a}`, :math:`\mathbf{b}`, and :math:`\mathbf{c}`
    must form a right-hand basis so that the box volume :math:`V` is positive:

    .. math::

        V = (\mathbf{a} \times \mathbf{b}) \cdot \mathbf{c} > 0

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
        If ``a``, ``b``, and ``c`` are not all 3-element vectors.
    ValueError
        If the volume is not positive.

    """
    def __init__(self, a, b, c):
        self.a = numpy.asarray(a,dtype=numpy.float64)
        self.b = numpy.asarray(b,dtype=numpy.float64)
        self.c = numpy.asarray(c,dtype=numpy.float64)
        if not (self.a.shape==(3,) and self.b.shape==(3,) and self.c.shape==(3,)):
            raise TypeError('a, b, and c must be 3-element vectors.')
        if self.volume <= 0:
            raise ValueError('The volume must be positive.')

    @property
    def volume(self):
        return numpy.dot(numpy.cross(self.a,self.b),self.c)

    def to_json(self):
        r"""Serialize as a dictionary.

        The dictionary contains the three box vectors ``a``, ``b``, and ``c`` as tuples.

        Returns
        -------
        dict
            The serialized Parallelepiped.

        """
        return {'a': tuple(self.a),
                'b': tuple(self.b),
                'c': tuple(self.c)
               }

    @classmethod
    def from_json(cls, data):
        r"""Deserialize from a dictionary.

        Parameters
        ----------
        data : dict
            The serialized equivalent of the Parallelepiped object. The keys
            of ``data`` should be ``('a','b','c')``, and the data for each
            is the 3-element box vector.

        Returns
        -------
        :class:`Parallelepiped`
            A new Parallelepiped object constructed from the data.

        """
        return Parallelepiped(**data)

class TriclinicBox(Parallelepiped):
    r"""Triclinic box.

    A TriclinicBox is a special type of :class:`Parallelepiped`. The box is
    defined by an orthorhombic box oriented along the Cartesian axes and having
    three vectors of length :math:`L_x`, :math:`L_y`, and :math:`L_z`, respectively.
    The box is then tilted by factors :math:`xy`, :math:`xz`, and :math:`yz`, which
    are upper off-diagonal elements of the matrix of box vectors. As a result,
    the :math:`\mathbf{a}` vector is always aligned along the :math:`x` axis, while
    the other two vectors may be tilted.

    The tilt factors can be defined using one of two :class:`TriclinicBox.Convention`\s.
    By default, the LAMMPS convention is applied to calculate the basis vectors.

    Parameters
    ----------
    Lx : float
        Length along the :math:`x` axis.
    Ly : float
        Length along the :math:`y` axis.
    Lz : float
        Length along the :math:`z` axis.
    xy : float
        First tilt factor.
    xz : float
        Second tilt factor.
    yz : float
        Third tilt factor.

    Raises
    ------
    ValueError
        If ``Lx``, ``Ly``, and ``Lz`` are not all positive.
    ValueError
        If the convention is not ``TriclinicBox.Convention.LAMMPS`` or
        ``TriclinicBox.Convention.HOOMD``.

    """

    class Convention(Enum):
        r"""Convention by which the tilt factors are applied to the basis vectors.

        In the `LAMMPS <https://lammps.sandia.gov/doc/Howto_triclinic.html>`_
        simulation convention, specified using ``TriclinicBox.Convention.LAMMPS``,
        the basis vectors are

        .. math::

            \mathbf{a} = (L_x,0,0)
            \quad \mathbf{b} = (xy,L_y,0)
            \quad \mathbf{c} = (xz,yz,L_z)

        In the `HOOMD-blue <https://hoomd-blue.readthedocs.io/en/stable/box.html>`_
        simulation convention, specified using ``TriclinicBox.Convention.HOOMD``,
        the basis vectors are

        .. math::

            \mathbf{a} = (L_x,0,0)
            \quad \mathbf{b} = (xy \cdot L_y,L_y,0)
            \quad \mathbf{c} = (xz \cdot L_z,yz \cdot L_z,L_z)

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
        if self.convention is TriclinicBox.Convention.LAMMPS:
            a = (Lx,0,0)
            b = (xy,Ly,0)
            c = (xz,yz,Lz)
        elif self.convention is TriclinicBox.Convention.HOOMD:
            a = (Lx,0,0)
            b = (xy*Ly,Ly,0)
            c = (xz*Lz,yz*Lz,Lz)
        else:
            raise ValueError('Triclinic convention must be TriclinicBox.Convention.LAMMPS or TriclinicBox.Convention.HOOMD')
        super().__init__(a,b,c)

    @property
    def convention(self):
        r""":class:`TriclinicBox.Convention`: Convention for tilt factors."""
        return self._convention

    def to_json(self):
        r"""Serialize as a dictionary.

        The dictionary contains the three box lengths ``Lx``, ``Ly``, and ``Lz``,
        the three tilt factors ``xy``, ``xz``, and ``yz``, and the ``convention``
        for the tilt factors.

        Returns
        -------
        dict
            The serialized TriclinicBox.

        """
        if self._convention is TriclinicBox.Convention.LAMMPS:
            xy = self.b[0]
            xz = self.c[0]
            yz = self.c[1]
        elif self._convention is TriclinicBox.Convention.HOOMD:
            xy = self.b[0]/self.b[1]
            xz = self.c[0]/self.c[2]
            yz = self.c[1]/self.c[2]
        return {'Lx': self.a[0],
                'Ly': self.b[1],
                'Lz': self.c[2],
                'xy': xy,
                'xz': xz,
                'yz': yz,
                'convention': self._convention.name
               }

    @classmethod
    def from_json(cls, data):
        r"""Deserialize from a dictionary.

        Parameters
        ----------
        data : dict
            The serialized equivalent of the TriclinicBox object. The keys
            of ``data`` should be ``('Lx','Ly','Lz','xy','xz','yz','convention')``.
            The lengths and tilt factors should be floats, and the convention should
            be a string.

        Returns
        -------
        :class:`TriclinicBox`
            A new TriclinicBox object constructed from the data.

        Raises
        ------
        ValueError
            If the convention specified is not ``'LAMMPS'`` or ``'HOOMD'``.

        """
        data_ = dict(data)
        if data['convention']=='LAMMPS':
            data_['convention'] = TriclinicBox.Convention.LAMMPS
        elif data['convention']=='HOOMD':
            data_['convention'] = TriclinicBox.Convention.HOOMD
        else:
            return ValueError('Only LAMMPS and HOOMD conventions are supported.')
        return TriclinicBox(**data_)

class Cuboid(TriclinicBox):
    r"""Orthorhombic box.

    A Cuboid is a special type of :class:`TriclinicBox`. The three box vectors
    point along the :math:`x`, :math:`y`, and :math:`z` axes, so they are all
    orthogonal (i.e. :math:`xy=xz=yz=0`). Each vector can have a different length,
    :math:`L_x`, :math:`L_y`, and :math:`L_z`.

    Parameters
    ----------
    Lx : float
        Length along the :math:`x` axis.
    Ly : float
        Length along the :math:`y` axis.
    Lz : float
        Length along the :math:`z` axis.

    """
    def __init__(self, Lx, Ly, Lz):
        super().__init__(Lx,Ly,Lz,0,0,0)

    def to_json(self):
        r"""Serialize as a dictionary.

        The dictionary contains the three box lengths ``Lx``, ``Ly``, and ``Lz``.

        Returns
        -------
        dict
            The serialized Cuboid.

        """
        return {'Lx': self.a[0],
                'Ly': self.b[1],
                'Lz': self.c[2],
               }

    @classmethod
    def from_json(cls, data):
        r"""Deserialize from a dictionary.

        Parameters
        ----------
        data : dict
            The serialized equivalent of the Cuboid object. The keys
            of ``data`` should be ``('Lx','Ly','Lz')``, and their values
            should be floats.

        Returns
        -------
        :class:`Cuboid`
            A new Cuboid object constructed from the data.

        """
        return Cuboid(**data)

class Cube(Cuboid):
    r"""Cubic box.

    A Cube is a special type of :class:`Cuboid` where all vectors have the
    same length :math:`L`.

    Parameters
    ----------
    L : float
        The edge length of the cube.

    """
    def __init__(self, L):
        super().__init__(L,L,L)

    def to_json(self):
        r"""Serialize as a dictionary.

        The dictionary contains the box length ``L``.

        Returns
        -------
        dict
            The serialized Cube.

        """
        return {'L': self.a[0]}

    @classmethod
    def from_json(cls, data):
        r"""Deserialize from a dictionary.

        Parameters
        ----------
        data : dict
            The serialized equivalent of the Cube object. The keys
            of ``data`` should be ``('L',)``, and its value should be a float.

        Returns
        -------
        :class:`Cube`
            A new Cube object constructed from the data.

        """
        return Cube(**data)

class Parallelegram(Area):
        r"""Parallelepiped box defined by three vectors.

    The three vectors :math:`\mathbf{a}`, :math:`\mathbf{b}`, and :math:`\mathbf{c}`
    must form a right-hand basis so that the box volume :math:`V` is positive:

    .. math::

        V = (\mathbf{a} \times \mathbf{b}) \cdot \mathbf{c} > 0

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
        If ``a``, ``b``, and ``c`` are not all 3-element vectors.
    ValueError
        If the volume is not positive.

    """
