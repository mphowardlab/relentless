"""
Extents
=======
An :class:`Extent` represents a region of space with a fixed, scalar volume or area.
It corresponds to the "box" used in simulations.

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

"""

import abc

import numpy


class Extent(abc.ABC):
    r"""Abstract base class defining a region of space.

    An Extent can be any region of space; typically, it is a simulation "box."
    Any deriving class must implement the extent method that computes the scalar
    size of the region. Additionally, methods to serialize and deserialize an
    Extent must be specified so that the object can be saved to disk.

    """

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


class Volume(Extent):
    r"""Three-dimensional region of space.

    A Volume can be any region of space; typically, it is a simulation "box."
    Any deriving class must implement the volume method that computes the scalar
    volume of the region. Additionally, methods to serialize and deserialize a
    Volume must be specified so that the object can be saved to disk.

    """

    pass


class TriclinicBox(Volume):
    r"""Triclinic box.

    A TriclinicBox is defined by three edge vectors **a**, **b**, and **c** that form
    a right-hand basis. These vectors are defined by deformation of an orthorhombic
    box oriented along the Cartesian axes and having three vectors of length
    :math:`L_x`, :math:`L_y`, and :math:`L_z`, respectively. The box is then
    tilted by factors :math:`xy`, :math:`xz`, and :math:`yz`, which
    are upper off-diagonal elements of the matrix of box vectors. As a result,
    the **a** vector is always aligned along the :math:`x` axis, while
    the other two vectors may be tilted.

    The box is centered at the origin :math:`(0,0,0)`.

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
    convention : {'LAMMPS', 'HOOMD'}
        Convention for defining the tilt factors.

    Notes
    -----
    .. rubric:: Convention

    The tilt factors can be defined using one of two conventions. In the
    `LAMMPS <https://lammps.sandia.gov/doc/Howto_triclinic.html>`_
    convention, the basis vectors are:

    .. math::

        \mathbf{a} = (L_x,0,0)
        \quad \mathbf{b} = (xy,L_y,0)
        \quad \mathbf{c} = (xz,yz,L_z)

    In the `HOOMD <https://hoomd-blue.readthedocs.io/en/stable/box.html>`_
    simulation convention, the basis vectors are:

    .. math::

        \mathbf{a} = (L_x,0,0)
        \quad \mathbf{b} = (xy \cdot L_y,L_y,0)
        \quad \mathbf{c} = (xz \cdot L_z,yz \cdot L_z,L_z)

    """

    def __init__(self, Lx, Ly, Lz, xy, xz, yz, convention="LAMMPS"):
        if Lx <= 0 or Ly <= 0 or Lz <= 0:
            raise ValueError("All side lengths must be positive.")

        convention = convention.upper()
        if convention == "LAMMPS":
            a = (Lx, 0, 0)
            b = (xy, Ly, 0)
            c = (xz, yz, Lz)
        elif convention == "HOOMD":
            a = (Lx, 0, 0)
            b = (xy * Ly, Ly, 0)
            c = (xz * Lz, yz * Lz, Lz)
        else:
            raise ValueError("Triclinic convention must be LAMMPS or HOOMD")

        self.a = numpy.array(a, dtype=float)
        self.b = numpy.array(b, dtype=float)
        self.c = numpy.array(c, dtype=float)

        box_vec = self.a + self.b + self.c
        self._low = -0.5 * box_vec
        self._high = 0.5 * box_vec
        self._extent = numpy.dot(numpy.cross(self.a, self.b), self.c)
        self._convention = convention

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
        return TriclinicBox(**data)

    @property
    def extent(self):
        return self._extent

    @property
    def low(self):
        return self._low

    @property
    def high(self):
        return self._high

    def as_array(self, convention=None):
        """Convert to array of lengths and tilt factors.

        Parameters
        ----------
        convention : {'LAMMPS','HOOMD'}, optional
            Convention to use for the tilt factors. Default of ``None``
            will use the convention for the box.

        Returns
        -------
        numpy.ndarray
            An array containing ``(Lx,Ly,Lz,xy,xz,yz)`` according to the
            ``convention``.

        """
        if convention is None:
            convention = self._convention
        convention = convention.upper()

        Lx = self.a[0]
        Ly = self.b[1]
        Lz = self.c[2]
        if convention == "LAMMPS":
            xy = self.b[0]
            xz = self.c[0]
            yz = self.c[1]
        elif convention == "HOOMD":
            xy = self.b[0] / self.b[1]
            xz = self.c[0] / self.c[2]
            yz = self.c[1] / self.c[2]
        else:
            raise TypeError("Convention must be LAMMPS or HOOMD")

        return numpy.array([Lx, Ly, Lz, xy, xz, yz])

    def coordinate_to_fraction(self, r):
        r"""Make fractional coordinates from Cartesian coordinates.

        The Cartesian coordinates **r** are projected onto the three
        (potentially nonorthogonal) basis vectors defining the box to yield
        fractional coordinates **x** such that:

        .. math::

            \mathbf{r} = \mathbf{r}_{\rm low}
                + (\mathbf{a}\quad\mathbf{b}\quad\mathbf{c}) \cdot \mathbf{x}

        where :math:`\mathbf{r}_{\rm low}` is the lower bound of the box,
        i.e., :attr:`low`.

        Parameters
        ----------
        r : array_like
            Cartesian coordinates (or array of).

        Returns
        -------
        numpy.ndarray
            Fractional coordinates **x** corresponding to **r**.

        """
        Lx, Ly, Lz, xy, xz, yz = self.as_array("LAMMPS")

        # get difference from lower bound
        dx = -self.low + r
        is_1d = False
        if len(dx.shape) == 1:
            dx = dx[numpy.newaxis, ...]
            is_1d = True

        # make fractional coordinate
        x = numpy.zeros_like(dx)
        x[:, 0] = (
            (1.0 / Lx) * dx[..., 0]
            + (-xy / (Lx * Ly)) * dx[..., 1]
            + ((yz * xy - Ly * xz) / (Lx * Ly * Lz)) * dx[..., 2]
        )
        x[:, 1] = (1.0 / Ly) * dx[..., 1] + (-yz / (Ly * Lz)) * dx[..., 2]
        x[:, 2] = (1.0 / Lz) * dx[..., 2]
        if is_1d:
            x = x[0]

        return x

    def fraction_to_coordinate(self, x):
        r"""Make Cartesian coordinates from fractional coordinates.

        The fractional coordinates **x** are converted to Cartesian coordinates **r**
        using the basis vectors of the box. See :meth:`coordinate_to_fraction`
        for the definition of these coordinates.

        Parameters
        ----------
        r : array_like
            Fractional coordinates (or array of).

        Returns
        -------
        numpy.ndarray
            Cartesian coordinates **r** corresponding to **x**.

        """
        Lx, Ly, Lz, xy, xz, yz = self.as_array("LAMMPS")

        # make real coordinate
        r = numpy.array(x)
        is_1d = False
        if len(r.shape) == 1:
            r = r[numpy.newaxis, ...]
            is_1d = True
        r[:, 0] = Lx * r[..., 0] + xy * r[..., 1] + xz * r[..., 2]
        r[:, 1] = Ly * r[..., 1] + yz * r[..., 2]
        r[:, 2] = Lz * r[..., 2]
        r += self.low
        if is_1d:
            r = r[0]

        return r

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
        Lx, Ly, Lz, xy, xz, yz = self.as_array()
        return {
            "Lx": float(Lx),
            "Ly": float(Ly),
            "Lz": float(Lz),
            "xy": float(xy),
            "xz": float(xz),
            "yz": float(yz),
            "convention": self._convention,
        }

    def wrap(self, positions):
        """Wrap positions subject to periodic boundary conditions.

        Three-dimensional periodic boundary conditions are applied to ensure
        the ``positions`` lie within the box. This is achieved by converting
        to fractional coordinates, bounding the fractional coordinates within
        :math:`[0,1)`, then converting back to Cartesian coordinates.

        Parameters
        ----------
        positions : array_like
            Position vector(s).

        Returns
        -------
        numpy.ndarray
            Wrapped position(s).

        """
        x = self.coordinate_to_fraction(positions)
        x -= numpy.floor(x)
        r = self.fraction_to_coordinate(x)
        return r


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
        super().__init__(Lx, Ly, Lz, 0, 0, 0)

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

    def to_json(self):
        r"""Serialize as a dictionary.

        The dictionary contains the three box lengths ``Lx``, ``Ly``, and ``Lz``.

        Returns
        -------
        dict
            The serialized Cuboid.

        """
        return {
            "Lx": float(self.a[0]),
            "Ly": float(self.b[1]),
            "Lz": float(self.c[2]),
        }


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
        super().__init__(L, L, L)

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

    def to_json(self):
        r"""Serialize as a dictionary.

        The dictionary contains the box length ``L``.

        Returns
        -------
        dict
            The serialized Cube.

        """
        return {"L": float(self.a[0])}


class Area(Extent):
    r"""Two-dimensional region of space.

    An Area can be any 2d region of space; typically, it is a simulation "box."
    Any deriving class must implement the volume method that computes the scalar
    area of the region. Additionally, methods to serialize and deserialize a
    Area must be specified so that the object can be saved to disk.

    """

    pass


class ObliqueArea(Area):
    r"""Oblique area.

    An ObliqueArea is defined by two edge vectors **a** and **b** that form
    a right-hand basis. These vectors are defined by deformation of a rectangle
    oriented along the Cartesian axes and having two edge lengths :math:`L_x`
    and :math:`L_y`, respectively. The box is then tilted by a factor :math:`xy`,
    which is the upper off-diagonal element of the matrix of box vectors. As a result,
    the **a** vector is always aligned along the :math:`x` axis, while
    the other vector may be tilted.

    Parameters
    ----------
    Lx : float
        Length along the :math:`x` axis.
    Ly : float
        Length along the :math:`y` axis.
    xy : float
        Tilt factor.
    convention : {'LAMMPS','HOOMD'}
        Convention for the tilt factor.

    Notes
    -----
    .. rubric:: Convention

    The tilt factors can be defined using one of two conventions. In the
    `LAMMPS <https://lammps.sandia.gov/doc/Howto_triclinic.html>`_
    convention, the basis vectors are:

    .. math::

        \mathbf{a} = (L_x,0,0)
        \quad \mathbf{b} = (xy,L_y,0)

    In the `HOOMD <https://hoomd-blue.readthedocs.io/en/stable/box.html>`_
    simulation convention, the basis vectors are:

    .. math::

        \mathbf{a} = (L_x,0,0)
        \quad \mathbf{b} = (xy \cdot L_y,L_y,0)

    """

    def __init__(self, Lx, Ly, xy, convention="LAMMPS"):
        if Lx <= 0 or Ly <= 0:
            raise ValueError("All side lengths must be positive.")

        convention = convention.upper()
        if convention == "LAMMPS":
            a = (Lx, 0)
            b = (xy, Ly)
        elif convention == "HOOMD":
            a = (Lx, 0)
            b = (xy * Ly, Ly)
        else:
            raise ValueError("Triclinic convention must be LAMMPS or HOOMD")

        self.a = numpy.asarray(a, dtype=float)
        self.b = numpy.asarray(b, dtype=float)

        box_vec = self.a + self.b
        self._low = -0.5 * box_vec
        self._high = 0.5 * box_vec
        self._extent = numpy.linalg.det([self.a, self.b])
        self._convention = convention

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

        """
        return ObliqueArea(**data)

    @property
    def extent(self):
        return self._extent

    @property
    def low(self):
        return self._low

    @property
    def high(self):
        return self._high

    def as_array(self, convention=None):
        """Convert to array of lengths and tilt factor.

        Parameters
        ----------
        convention : {'LAMMPS','HOOMD'}, optional
            Convention to use for the tilt factors. Default of ``None``
            will use the convention for the box.

        Returns
        -------
        numpy.ndarray
            An array containing ``(Lx,Ly,xy)`` according to the
            ``convention``.

        """
        if convention is None:
            convention = self._convention
        convention = convention.upper()

        Lx = self.a[0]
        Ly = self.b[1]
        if convention == "LAMMPS":
            xy = self.b[0]
        elif convention == "HOOMD":
            xy = self.b[0] / self.b[1]
        else:
            raise TypeError("Convention must be HOOMD or LAMMPS")
        return numpy.array([Lx, Ly, xy])

    def coordinate_to_fraction(self, r):
        r"""Make fractional coordinates from Cartesian coordinates.

        The Cartesian coordinates **r** are projected onto the two
        (potentially nonorthogonal) basis vectors defining the box to yield
        fractional coordinates **x** such that:

        .. math::

            \mathbf{r} = \mathbf{r}_{\rm low}
                + (\mathbf{a}\quad\mathbf{b}) \cdot \mathbf{x}

        where :math:`\mathbf{r}_{\rm low}` is the lower bound of the box,
        i.e., :attr:`low`.

        Parameters
        ----------
        r : array_like
            Cartesian coordinates (or array of).

        Returns
        -------
        numpy.ndarray
            Fractional coordinates **x** corresponding to **r**.

        """
        Lx, Ly, xy = self.as_array("LAMMPS")

        # get difference from lower bound
        dx = -self.low + r
        is_1d = False
        if len(dx.shape) == 1:
            dx = dx[numpy.newaxis, ...]
            is_1d = True

        # make fractional coordinate
        x = numpy.zeros_like(dx)
        x[:, 0] = (1.0 / Lx) * dx[..., 0] + (-xy / (Lx * Ly)) * dx[..., 1]
        x[:, 1] = (1.0 / Ly) * dx[..., 1]
        if is_1d:
            x = x[0]

        return x

    def fraction_to_coordinate(self, x):
        r"""Make Cartesian coordinates from fractional coordinates.

        The fractional coordinates **x** are converted to Cartesian coordinates **r**
        using the basis vectors of the box. See :meth:`coordinate_to_fraction`
        for the definition of these coordinates.

        Parameters
        ----------
        r : array_like
            Fractional coordinates (or array of).

        Returns
        -------
        numpy.ndarray
            Cartesian coordinates **r** corresponding to **x**.

        """
        Lx, Ly, xy = self.as_array("LAMMPS")

        # make real coordinate
        r = numpy.array(x)
        is_1d = False
        if len(r.shape) == 1:
            r = r[numpy.newaxis, ...]
            is_1d = True
        r[:, 0] = Lx * r[..., 0] + xy * r[..., 1]
        r[:, 1] = Ly * r[..., 1]
        r += self.low
        if is_1d:
            r = r[0]

        return r

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
        Lx, Ly, xy = self.as_array()
        return {
            "Lx": float(Lx),
            "Ly": float(Ly),
            "xy": float(xy),
            "convention": self._convention,
        }

    def wrap(self, positions):
        """Wrap positions subject to periodic boundary conditions.

        Two-dimensional periodic boundary conditions are applied to ensure
        the ``positions`` lie within the box. This is achieved by converting
        to fractional coordinates, bounding the fractional coordinates within
        :math:`[0,1)`, then converting back to Cartesian coordinates.

        Parameters
        ----------
        positions : array_like
            Position vector(s).

        Returns
        -------
        numpy.ndarray
            Wrapped position(s).

        """
        x = self.coordinate_to_fraction(positions)
        x -= numpy.round(x - 0.5)
        r = self.fraction_to_coordinate(x)
        return r


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
        super().__init__(Lx, Ly, 0)

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

    def to_json(self):
        r"""Serialize as a dictionary.

        The dictionary contains the two box lengths ``Lx`` and ``Ly``.

        Returns
        -------
        dict
            The serialized Rectangle.

        """
        return {
            "Lx": float(self.a[0]),
            "Ly": float(self.b[1]),
        }


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
        super().__init__(L, L)

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

    def to_json(self):
        r"""Serialize as a dictionary.

        The dictionary contains the box length ``L``.

        Returns
        -------
        dict
            The serialized Square.

        """
        return {"L": float(self.a[0])}
