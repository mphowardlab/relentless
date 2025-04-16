import numpy

from . import potential


class DihedralParameters(potential.Parameters):
    """Parameters for a dihedral potential."""

    pass


class DihedralPotential(potential.BondedPotential):
    r"""Abstract base class for an dihedral potential.

    The dihedral between four bonded particles :math:`i`, :math:`j`, :math:`k`,
    :math:`\ell`,  is defined by the angle between two planes :math:`ijk` and
    :math:`jk\ell`:

    .. math::

        \phi_{ijk\ell} = \arccos\left(\frac{\mathbf{n}_{A} \cdot
        \mathbf{n}_{B}}{|\mathbf{n}_{A}||\mathbf{r}_{B}|}\right)

    where :math:`\mathbf{n}_{A}` and :math:`\mathbf{n}_{B}` are the normal
    vectors to the planes :math:`ijk` and :math:`jk\ell`:

    .. math::

        \mathbf{n}_{A} = \mathbf{r}_{ij} \times \mathbf{r}_{jk}

    and

    .. math::

        \mathbf{n}_{B} = \mathbf{r}_{jk} \times \mathbf{r}_{k\ell}

    where :math:`\mathbf{r}_{ij}`,  :math:`\mathbf{r}_{jk}`, :math:`\mathbf{r}_{k\ell}`
    are the vectors from particle *i* to particle *j*, from particle *j* to
    particle *k*, and particle *k* to particle :math:`\ell`, respectively.
    :math:`\phi` is bound between :math:`-\pi` and :math:`\pi`.

    """

    def derivative(self, type_, var, phi):
        r"""Evaluate derivative with respect to a variable.

        The derivative is evaluated using the :meth:`_derivative` function for all
        :math:`u_{0,\lambda}(phi)`.

        The derivative will be carried out with respect to ``var`` for all
        :class:`~relentless.variable.Variable` parameters. The appropriate chain
        rules are handled automatically. If the potential does not depend on
        ``var``, the derivative will be zero by definition.

        Parameters
        ----------
        _type : tuple[str]
            The type for which to calculate the derivative.
        var : :class:`~relentless.variable.Variable`
            The variable with respect to which the derivative is calculated.
        phi : float or list
            The dihedral angles(s) at which to evaluate the derivative.

        Returns
        -------
        float or numpy.ndarray
            The dihedral derivative evaluated at ``phi``. The return type is
            consistent with ``phi``.

        Raises
        ------
        ValueError
            If any value in ``phi`` is not between -pi and pi.
        TypeError
            If the parameter with respect to which to take the derivative
            is not a :class:`~relentless.variable.Variable`.

        """
        self._validate_coordinate(phi)

        return super().derivative(type_=type_, var=var, x=phi)

    def _validate_coordinate(self, phi):
        """Validate the angle ``phi`` is between -pi and pi.

        Parameters
        ----------
        phi : float or list
            The dihedral(s) to validate.

        Raises
        ------
        ValueError
            If any value in ``phi`` is not between -pi and pi.
        """
        if numpy.any(numpy.less(phi, -numpy.pi)) or numpy.any(
            numpy.greater(phi, numpy.pi)
        ):
            raise ValueError("Angle must be between -pi and pi.")


class OPLSDihedral(DihedralPotential):
    r"""OPLS dihedral potential.

    .. math::

        u(\phi) = \frac{1}{2} \left( k_1 (1+\cos \phi) + k_2 (1+\cos 2\phi)
        + k_3 (1+ \cos 3\phi) + k_4 (1+ \cos 4\phi) \right)

    where :math:`\phi` is the dihedral between four bonded particles. The potential
    is described in `Watkins and Jorgensen`_.  The parameters for each type are:

    +-------------+--------------------------------------------------+
    | Parameter   | Description                                      |
    +=============+==================================================+
    | ``k_1``     | First fitting coefficient :math:`k_1`.           |
    +-------------+--------------------------------------------------+
    | ``k_2``     | Second fitting coefficient :math:`k_2`.          |
    +-------------+--------------------------------------------------+
    | ``k_3``     | Third fitting coefficient :math:`k_3`.           |
    +-------------+--------------------------------------------------+
    | ``k_4``     | Fourth fitting coefficient :math:`k_4`.          |
    +-------------+--------------------------------------------------+

    Parameters
    ----------
    types : tuple[str]
        Types.
    name : str
        Unique name of the potential. Defaults to ``__u[id]``, where ``id`` is the
        unique integer ID of the potential.

    Attributes
    ----------
    coeff : :class:`DihedralParameters`
        Parameters of the potential for each type.

    Examples
    --------
    OPLS dihedral for the CT-CT-CT-CT dihedral (`Watkins and Jorgensen`_). ::

        >>> u = relentless.potential.dihedral.OPLSDihedral(("A",))
        >>> u.coeff["A"].update({'k1': 1.740, 'k2': -0.157, 'k3': 0.279, 'k4': 0.00})

    .. _Watkins and Jorgensen: https://doi.org/10.1021/jp004071w

    """

    def __init__(self, types, name=None):
        super().__init__(keys=types, params=("k1", "k2", "k3", "k4"), name=name)

    def energy(self, type_, phi):
        """Evaluate dihedral energy."""
        params = self.coeff.evaluate(type_)
        k1 = params["k1"]
        k2 = params["k2"]
        k3 = params["k3"]
        k4 = params["k4"]

        phi, u, s = self._zeros(phi)

        self._validate_coordinate(phi)

        u = 0.5 * (
            k1 * (1 + numpy.cos(phi))
            + k2 * (1 - numpy.cos(2 * phi))
            + k3 * (1 + numpy.cos(3 * phi))
            + k4 * (1 - numpy.cos(4 * phi))
        )

        if s:
            u = u.item()
        return u

    def force(self, type_, phi):
        """Evaluate dihedral force."""
        params = self.coeff.evaluate(type_)
        k1 = params["k1"]
        k2 = params["k2"]
        k3 = params["k3"]
        k4 = params["k4"]

        phi, f, s = self._zeros(phi)

        self._validate_coordinate(phi)

        f = -0.5 * (
            -k1 * numpy.sin(phi)
            + 2 * k2 * numpy.sin(2 * phi)
            - 3 * k3 * numpy.sin(3 * phi)
            + 4 * k4 * numpy.sin(4 * phi)
        )

        if s:
            f = f.item()
        return f

    def _derivative(self, param, phi, k1, k2, k3, k4):
        r"""Evaluate dihedral derivative with respect to a variable."""
        phi, d, s = self._zeros(phi)

        if param == "k1":
            d = 0.5 * (1 + numpy.cos(phi))
        elif param == "k2":
            d = -0.5 * (1 + numpy.cos(2 * phi))
        elif param == "k3":
            d = 0.5 * (1 + numpy.cos(3 * phi))
        elif param == "k4":
            d = -0.5 * (1 + numpy.cos(4 * phi))
        else:
            raise ValueError("Unknown parameter")

        if s:
            d = d.item()
        return d


class RyckaertBellemansDihedral(DihedralPotential):
    r"""Ryckaert-Bellemans dihedral potential.

    .. math::

        u(\phi) = c_0 + c_1 (\cos (\phi - \pi)) + c_2 (\cos (\phi - \pi))^2 +
        c_3 (\cos (\phi - \pi))^3 + c_4 (\cos (\phi - \pi))^4 +
        c_5 (\cos (\phi - \pi))^5

    where :math:`\phi` is the dihedral between four bonded particles. The potential
    is described in `Ryckaert and Bellemans`_. The parameters for each type are:

    +-------------+--------------------------------------------------+
    | Parameter   | Description                                      |
    +=============+==================================================+
    | ``c_0``     | First fitting coefficient :math:`c_0`.           |
    +-------------+--------------------------------------------------+
    | ``c_1``     | Second fitting coefficient :math:`c_1`.          |
    +-------------+--------------------------------------------------+
    | ``c_2``     | Third fitting coefficient :math:`c_2`.           |
    +-------------+--------------------------------------------------+
    | ``c_3``     | Fourth fitting coefficient :math:`c_3`.          |
    +-------------+--------------------------------------------------+
    | ``c_4``     | Fifth fitting coefficient :math:`c_4`.           |
    +-------------+--------------------------------------------------+
    | ``c_5``     | Sixth fitting coefficient :math:`c_5`.           |
    +-------------+--------------------------------------------------+

    Parameters
    ----------
    types : tuple[str]
        Types.
    name : str
        Unique name of the potential. Defaults to ``__u[id]``, where ``id`` is the
        unique integer ID of the potential.

    Attributes
    ----------
    coeff : :class:`DihedralParameters`
        Parameters of the potential for each type.

    Examples
    --------
    Ryckaert-Bellemans Dihedral for :math:`\rm{CH}_2\rm{CH}_2\rm{CH}_2\rm{CH}_2`
    and :math:`\rm{CH}_2\rm{CH}_2\rm{CH}_2\rm{CH}_3` (`van Buuren et al.`_)::

        >>> u = relentless.potential.dihedral.RyckaertBellemansDihedral(("A",))
        >>> u.coeff["A"].update({
            'c0': 9.28,
            'c1': 12.16,
            'c2': -13.12,
            'c3': -3.06,
            'c4': 26.24,
            'c5': -31.5
            })

    .. _Ryckaert and Bellemans: https://doi.org/10.1039/DC9786600095
    .. _van Buuren et al.: https://doi.org/10.1021/j100138a023
    """

    def __init__(self, types, name=None):
        super().__init__(
            keys=types, params=("c0", "c1", "c2", "c3", "c4", "c5"), name=name
        )

    def energy(self, type_, phi):
        """Evaluate dihedral energy."""
        params = self.coeff.evaluate(type_)
        c0 = params["c0"]
        c1 = params["c1"]
        c2 = params["c2"]
        c3 = params["c3"]
        c4 = params["c4"]
        c5 = params["c5"]

        phi, u, s = self._zeros(phi)

        self._validate_coordinate(phi)

        psi = phi - numpy.pi
        cos_psi = numpy.cos(psi)

        u = (
            c0
            + c1 * cos_psi
            + c2 * cos_psi**2
            + c3 * cos_psi**3
            + c4 * cos_psi**4
            + c5 * cos_psi**5
        )

        if s:
            u = u.item()
        return u

    def force(self, type_, phi):
        """Evaluate dihedral force."""
        params = self.coeff.evaluate(type_)
        c1 = params["c1"]
        c2 = params["c2"]
        c3 = params["c3"]
        c4 = params["c4"]

        phi, f, s = self._zeros(phi)

        self._validate_coordinate(phi)

        psi = phi - numpy.pi
        cos_psi = numpy.cos(psi)
        sin_psi = numpy.sin(psi)

        f = (
            c1 * sin_psi
            + 2 * c2 * cos_psi * sin_psi
            + 3 * c3 * cos_psi**2 * sin_psi
            + 4 * c4 * cos_psi**3 * sin_psi
        )
        if s:
            f = f.item()
        return f

    def _derivative(self, param, phi, c0, c1, c2, c3, c4, c5):
        r"""Evaluate dihedral derivative with respect to a variable."""
        phi, d, s = self._zeros(phi)

        psi = phi - numpy.pi

        if param == "c0":
            d = numpy.ones_like(psi)
        elif param == "c1":
            d = numpy.cos(psi)
        elif param == "c2":
            d = numpy.cos(psi) ** 2
        elif param == "c3":
            d = numpy.cos(psi) ** 3
        elif param == "c4":
            d = numpy.cos(psi) ** 4
        elif param == "c5":
            d = numpy.cos(psi) ** 5
        else:
            raise ValueError("Unknown parameter")

        if s:
            d = d.item()
        return d


class DihedralSpline(potential.BondedSpline, DihedralPotential):
    """Spline dihedral potentials.

    The dihedral spline potential is defined by interpolation through a set of
    knot points. The interpolation scheme uses Akima splines.

    Parameters
    ----------
    types : tuple[str]
        Types.
    num_knots : int
        Number of knots.
    mode : str
        Mode for storing the values of the knots in
        :class:`~relentless.variable.Variable` that can be optimized. If
        ``mode='value'``, the knot amplitudes are manipulated directly.
        If ``mode='diff'``, the amplitude of the *last* knot is fixed, and
        differences between neighboring knots are manipulated for all other
        knots. Defaults to ``'diff'``.
    name : str
        Unique name of the potential. Defaults to ``__u[id]``, where ``id`` is the
        unique integer ID of the potential.

    Examples
    --------
    The spline potential is setup from a tabulated potential instead of
    specifying knot parameters directly::

        spline = relentless.potential.dihedral.DihedralSpline(
            types=("dihedralA",),
            num_knots=3,
        )
        spline.from_array(("dihedralA"),[0,1,2],[4,2,0])

    However, the knot variables can be iterated over and manipulated directly::

        for r,k in spline.knots("dihedralA"):
            k.value = 1.0

    """

    _space_coord_name = "phi"

    def from_array(self, types, phi, u):
        r"""Set up the potential from knot points.

        Parameters
        ----------
        types : tuple[str]
            The type for which to set up the potential.
        phi : list
            Position of each knot.
        u : list
            Potential energy of each knot.

        Raises
        ------
        ValueError
            If the number of ``phi`` values is not the same as the number of knots.
        ValueError
            If the number of ``u`` values is not the same as the number of knots.

        """

        if phi[0] != -numpy.pi and phi[-1] != numpy.pi:
            raise ValueError("The first and last knot must be at -pi and pi.")
        return super().from_array(types=types, x=phi, u=u)

    def energy(self, type_, phi):
        """Evaluate potential energy.

        Parameters
        ----------
        type_
            Type parametrizing the potential in :attr:`coeff<container>`.
        phi : float or list
            Potential energy coordinate.

        Returns
        -------
        float or numpy.ndarray
            The pair energy evaluated at ``phi``. The return type is consistent
            with ``phi``.

        Raises
        ------
        ValueError
            If any value in ``phi`` is outside of -pi to pi.
        """
        self._validate_coordinate(phi)
        return super().energy(type_=type_, x=phi)

    def force(self, type_, phi):
        """Evaluate force magnitude.

        The force is the (negative) magnitude of the ``phi`` gradient.

        Parameters
        ----------
        type_
            Type parametrizing the potential in :attr:`coeff<container>`.
        phi : float or list
            Potential energy coordinate.

        Returns
        -------
        float or numpy.ndarray
            The force evaluated at ``phi``. The return type is consistent
            with ``phi``.

        Raises
        ------
        ValueError
            If any value in ``phi`` is outside of -pi to pi.
        """
        self._validate_coordinate(phi)
        return super().force(type_=type_, x=phi)
