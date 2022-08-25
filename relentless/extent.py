"""
Extents
=======
An :class:`Extent` represents a region of space with a fixed, scalar volume or area. It corresponds
to the "box" used in simulations. 

The following three-dimensional box types have been implemented:

.. autosummary::
    :nosignatures:

    Parallelepiped
    TriclinicBox
    Cuboid
    Cube

The following two-dimensional box types have been implemented:

.. autosummary::
    :nosignatures:

    Parallelogram
    ObliqueArea
    Rectangle
    Square

The :class:`TriclinicBox` can be constructed using both the LAMMPS and HOOMD-blue
:class:`TriclinicBox.Convention`\s for applying tilt factors.

Examples
--------
Construct a simulation box with defined basis vectors and volume::

    v = relentless.extent.Cube(L=3)
    >>> print(v.a)
    [3.0 0.0 0.0]
    >>> print(v.b)
    [0.0 3.0 0.0]
    >>> print(v.c)
    [0.0 0.0 3.0]
    >>> print(v.extent)
    27.0

.. rubric:: Developer notes

To implement your own simulation box, create a class that derives from :class:`Volume`
and define the required methods.

.. autosummary::
    :nosignatures:

    Extent
    Volume
    Area

.. autoclass:: Extent
    :members:
.. autoclass:: Volume
    :members:
.. autoclass:: Area
    :members:
.. autoclass:: Parallelepiped
    :members:
.. autoclass:: TriclinicBox
    :members:
.. autoclass:: Cuboid
    :members:
.. autoclass:: Cube
    :members:
.. autoclass:: Parallelogram
    :members:
.. autoclass:: ObliqueArea
    :members:
.. autoclass:: Rectangle
    :members:
.. autoclass:: Square
    :members:

"""
import abc
from enum import Enum

import numpy

class Extent(abc.ABC):
    r"""Abstract base class defining a region of space.

    An Extent can be any region of space; typically, it is a simulation "box."
    Any deriving class must implement the extent method that computes the scalar
    size of the region. Additionally, methods to serialize and deserialize an
    Extent must be specified so that the object can be saved to disk.

    """
    @property
    @abc.abstractmethod
    def extent(self):
        r"""float: Extent of the region."""
        pass
    
    @abc.abstractmethod
    def to_json(self):
        r"""Serialize as a dictionary.

        The serialized data can be saved to file as JSON data.

        Returns
        -------
        dict
            The serialized data.

        """
        pass

    @classmethod
    @abc.abstractmethod
    def from_json(cls, data):
        r"""Deserialize from a dictionary.

        Returns
        -------
        :class:`Extent`
            The object reconstructed from the ``data``.

        """
        pass

class Volume(Extent):
    r"""Three-dimensional region of space.

    A Volume can be any region of space; typically, it is a simulation "box."
    Any deriving class must implement the volume method that computes the scalar
    volume of the region. Additionally, methods to serialize and deserialize a
    Volume must be specified so that the object can be saved to disk.

    """
    @property
    def volume(self):
        r"""float: Volume of the region."""
        pass

class Area(Extent):
    r"""Two-dimensional region of space.

    An Area can be any 2d region of space; typically, it is a simulation "box."
    Any deriving class must implement the volume method that computes the scalar
    area of the region. Additionally, methods to serialize and deserialize a
    Area must be specified so that the object can be saved to disk.

    """
    @property
    def area(self):
        r"""float: Area of the region."""
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
        if self.extent <= 0:
            raise ValueError('The volume must be positive.')

    @property
    def extent(self):
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

class Parallelogram(Area):
    r"""Parallelogram box defined by two vectors.

    The two vectors :math:`\mathbf{a}` and :math:`\mathbf{b}`
    must form a right-hand basis so that the box volume :math:`A` is positive:

    .. math::

        A = |(\mathbf{a} \times \mathbf{b})| > 0

    Parameters
    ----------
    a : array_like
        First vector defining the parallelogram.
    b : array_like
        Second vector defining the parallelogram.

    Raises
    ------
    TypeError
        If ``a`` and ``b`` are not both 2-element vectors.
    ValueError
        If the area is not positive.

    """
    def __init__(self, a, b):
        self.a = numpy.asarray(a,dtype=numpy.float64)
        self.b = numpy.asarray(b,dtype=numpy.float64)
        if not (self.a.shape==(2,) and self.b.shape==(2,)):
            raise TypeError('a and b must be 2-element vectors.')
        if self.extent <= 0:
            raise ValueError('The area must be positive.')

    @property
    def extent(self):
        return numpy.linalg.norm(numpy.cross(self.a,self.b))

    def to_json(self):
        r"""Serialize as a dictionary.

        The dictionary contains the two box vectors ``a`` and ``b`` as tuples.

        Returns
        -------
        dict
            The serialized Parallelogram.

        """
        return {'a': tuple(self.a),
                'b': tuple(self.b)
               }

    @classmethod
    def from_json(cls, data):
        r"""Deserialize from a dictionary.

        Parameters
        ----------
        data : dict
            The serialized equivalent of the Parallelogram object. The keys
            of ``data`` should be ``('a','b')``, and the data for each
            is the 2-element box vector.

        Returns
        -------
        :class:`Parallelogram`
            A new Parallelogram object constructed from the data.

        """
        return Parallelogram(**data)

class ObliqueArea(Parallelogram):
    r"""Oblique area.

    A ObliqueArea is a special type of :class:`Parallelogram`. The box is
    defined by an area oriented along the Cartesian axes and having
    two vectors of length :math:`L_x` and :math:`L_y`, respectively.
    The box is then tilted by factor :math:`xy`, which
    is upper off-diagonal elements of the matrix of box vectors. As a result,
    the :math:`\mathbf{a}` vector is always aligned along the :math:`x` axis, while
    the other vector may be tilted.

    The tilt factors can be defined using one of two :class:`ObliqueArea.Convention`\s.
    By default, the LAMMPS convention is applied to calculate the basis vectors.

    Parameters
    ----------
    Lx : float
        Length along the :math:`x` axis.
    Ly : float
        Length along the :math:`y` axis.
    xy : float
        Tilt factor.


    Raises
    ------
    ValueError
        If ``Lx`` and ``Ly`` are not both positive.
    ValueError
        If the convention is not ``ObliqueArea.Convention.LAMMPS`` or
        ``ObliqueArea.Convention.HOOMD``.

    """
    
    class Convention(Enum):
        r"""Convention by which the tilt factors are applied to the basis vectors.

        In the `LAMMPS <https://lammps.sandia.gov/doc/Howto_triclinic.html>`_
        simulation convention, specified using ``TriclinicBox.Convention.LAMMPS``,
        the basis vectors are

        .. math::

            \mathbf{a} = (L_x,0,0)
            \quad \mathbf{b} = (xy,L_y,0)

        In the `HOOMD-blue <https://hoomd-blue.readthedocs.io/en/stable/box.html>`_
        simulation convention, specified using ``TriclinicBox.Convention.HOOMD``,
        the basis vectors are

        .. math::

            \mathbf{a} = (L_x,0,0)
            \quad \mathbf{b} = (xy \cdot L_y,L_y,0)

        Attributes
        ----------
        LAMMPS : int
            LAMMPS convention for applying the tilt factors.
        HOOMD : int
            HOOMD convention for applying the tilt factors.

        """
        LAMMPS = 1
        HOOMD = 2

    def __init__(self, Lx, Ly, xy, convention=Convention.LAMMPS):
        if Lx<=0 or Ly<=0:
            raise ValueError('All side lengths must be positive.')
        self._convention = convention
        if self.convention is ObliqueArea.Convention.LAMMPS:
            a = (Lx,0)
            b = (xy,Ly)
        elif self.convention is ObliqueArea.Convention.HOOMD:
            a = (Lx,0)
            b = (xy*Ly,Ly)
        else:
            raise ValueError('Triclinic convention must be ObliqueArea.Convention.LAMMPS or ObliqueArea.Convention.HOOMD')
        super().__init__(a,b)

    @property
    def convention(self):
        r""":class:`ObliqueArea.Convention`: Convention for tilt factors."""
        return self._convention

    def to_json(self):
        r"""Serialize as a dictionary.

        The dictionary contains the two box lengths ``Lx`` and ``Ly``,
        the tilt factor ``xy``, and the ``convention``
        for the tilt factors.

        Returns
        -------
        dict
            The serialized ObliqueArea.

        """
        if self._convention is ObliqueArea.Convention.LAMMPS:
            xy = self.b[0]
        elif self._convention is ObliqueArea.Convention.HOOMD:
            xy = self.b[0]/self.b[1]
        return {'Lx': self.a[0],
                'Ly': self.b[1],
                'xy': xy,
                'convention': self._convention.name
               }

    @classmethod
    def from_json(cls, data):
        r"""Deserialize from a dictionary.

        Parameters
        ----------
        data : dict
            The serialized equivalent of the ObliqueArea object. The keys
            of ``data`` should be ``('Lx','Ly','xy','convention')``.
            The lengths and tilt factor should be floats, and the convention should
            be a string.

        Returns
        -------
        :class:`ObliqueArea`
            A new ObliqueArea object constructed from the data.

        Raises
        ------
        ValueError
            If the convention specified is not ``'LAMMPS'`` or ``'HOOMD'``.

        """
        data_ = dict(data)
        if data['convention']=='LAMMPS':
            data_['convention'] = ObliqueArea.Convention.LAMMPS
        elif data['convention']=='HOOMD':
            data_['convention'] = ObliqueArea.Convention.HOOMD
        else:
            return ValueError('Only LAMMPS and HOOMD conventions are supported.')
        return ObliqueArea(**data_)

class Rectangle(ObliqueArea):
    r"""Rectangle box.

    A Reactangle is a special type of :class:`ObliqueArea`. The two box vectors
    point along the :math:`x` and :math:`y`, so they are all
    orthogonal (i.e. :math:`xy=0`). Each vector can have a different length,
    :math:`L_x` and :math:`L_y`.

    Parameters
    ----------
    Lx : float
        Length along the :math:`x` axis.
    Ly : float
        Length along the :math:`y` axis..

    """
    def __init__(self, Lx, Ly):
        super().__init__(Lx,Ly,0)

    def to_json(self):
        r"""Serialize as a dictionary.

        The dictionary contains the two box lengths ``Lx`` and ``Ly``.

        Returns
        -------
        dict
            The serialized Rectangle.

        """
        return {'Lx': self.a[0],
                'Ly': self.b[1],
               }

    @classmethod
    def from_json(cls, data):
        r"""Deserialize from a dictionary.

        Parameters
        ----------
        data : dict
            The serialized equivalent of the Rectangle object. The keys
            of ``data`` should be ``('Lx','Ly')``, and their values
            should be floats.

        Returns
        -------
        :class:`Rectangle`
            A new Rectangle object constructed from the data.

        """
        return Rectangle(**data)

class Square(Rectangle):
    r"""Square box.

    A Square is a special type of :class:`Rectangle` where all vectors have the
    same length :math:`L`.

    Parameters
    ----------
    L : float
        The edge length of the square.

    """
    def __init__(self, L):
        super().__init__(L,L)

    def to_json(self):
        r"""Serialize as a dictionary.

        The dictionary contains the box length ``L``.

        Returns
        -------
        dict
            The serialized Square.

        """
        return {'L': self.a[0]}

    @classmethod
    def from_json(cls, data):
        r"""Deserialize from a dictionary.

        Parameters
        ----------
        data : dict
            The serialized equivalent of the Square object. The keys
            of ``data`` should be ``('L',)``, and its value should be a float.

        Returns
        -------
        :class:`Square`
            A new Square object constructed from the data.

        """
        return Square(**data)
