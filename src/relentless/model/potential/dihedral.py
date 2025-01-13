import numpy

from relentless.model.potential.bond import BondSpline

from . import potential


class DihedralParameters(potential.Parameters):
    """Parameters for a dihedral potential."""

    pass


class DihedralPotential(potential.BondedPotential):
    r"""Abstract base class for an dihedral potential."""

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
            The bond distance(s) at which to evaluate the derivative.

        Returns
        -------
        float or numpy.ndarray
            The bond derivative evaluated at ``phi``. The return type is consistent
            with ``phi``.

        Raises
        ------
        ValueError
            If any value in ``phi`` is negative.
        TypeError
            If the parameter with respect to which to take the derivative
            is not a :class:`~relentless.variable.Variable`.

        """

        return super().derivative(type_=type_, var=var, x=phi)

    pass


class OPLSDihedral(DihedralPotential):
    r"""OPLS dihedral potential.

    .. math::

        u(\phi) = \frac{1}{2} \left( k_1 (1+\cos \phi) + k_2 (1+\cos 2\phi)
        + k_3 (1+ \cos 3\phi) + k_4 (1+ \cos 4\phi) \right)

    where :math:`\phi` is the dihedral between four bonded particles. The parameters
    for each type are:

    +-------------+--------------------------------------------------+
    | Parameter   | Description                                      |
    +=============+==================================================+
    | ``k_1``     | First fitting coefficient :math:`k`.             |
    +-------------+--------------------------------------------------+
    | ``k_2``     | Second fitting coefficient :math:`k`.            |
    +-------------+--------------------------------------------------+
    | ``k_3``     | Third fitting coefficient :math:`k`.             |
    +-------------+--------------------------------------------------+
    | ``k_4``     | Fourth fitting coefficient :math:`k`.            |
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
    OPLS Dihedral::

        >>> u = relentless.potential.dihedral.OPLSDihedral(("A",))
        >>> u.coeff["A"].update({'k1': 1.740, 'k2': -0.157, 'k3': 0.279, 'k4': 0.00})

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
            d = 0.5 * (-1 - numpy.cos(2 * phi))
        elif param == "k3":
            d = 0.5 * (1 + numpy.cos(3 * phi))
        elif param == "k4":
            d = 0.5 * (-1 - numpy.cos(4 * phi))
        else:
            raise ValueError("Unknown parameter")

        if s:
            d = d.item()
        return d


class RyckaertBellemansDihedral(DihedralPotential):
    r"""Cosine squared dihedral potential.

    .. math::

        u(\phi) = c_0 + c_1 (\cos (\phi - \pi)) + c_2 (\cos (\phi - \pi))^2 +
        c_3 (\cos (\phi - \pi))^3 + c_4 (\cos (\phi - \pi))^4 +
        c_5 (\cos (\phi - \pi))^5

    where :math:`\phi` is the dihedral between four bonded particles. The parameters
    for each type are:

    +-------------+--------------------------------------------------+
    | Parameter   | Description                                      |
    +=============+==================================================+
    | ``c_0``     | First fitting coefficient :math:`k`.             |
    +-------------+--------------------------------------------------+
    | ``c_1``     | Second fitting coefficient :math:`k`.            |
    +-------------+--------------------------------------------------+
    | ``c_2``     | Third fitting coefficient :math:`k`.             |
    +-------------+--------------------------------------------------+
    | ``c_3``     | Fourth fitting coefficient :math:`k`.            |
    +-------------+--------------------------------------------------+
    | ``c_4``     | Fifth fitting coefficient :math:`k`.             |
    +-------------+--------------------------------------------------+
    | ``c_5``     | Sixth fitting coefficient :math:`k`.             |
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
    Harmonic Dihedral::

        >>> u = relentless.potential.dihedral.RyckaertBellemansDihedral(("A",))
        >>> u.coeff["A"].update({
            'c0': 9.28,
            'c1': 12.16,
            'c2': -13.12,
            'c3': -3.06,
            'c4': 26.24,
            'c5': -31.5
            })

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

        psi = phi - numpy.pi

        u = (
            c0
            + c1 * numpy.cos(psi)
            + c2 * numpy.cos(psi) ** 2
            + c3 * numpy.cos(psi) ** 3
            + c4 * numpy.cos(psi) ** 4
            + c5 * numpy.cos(psi) ** 5
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

        psi = phi - numpy.pi

        f = (
            c1 * numpy.sin(psi)
            + 2 * c2 * numpy.cos(psi) * numpy.sin(psi)
            + 3 * c3 * numpy.cos(psi) ** 2 * numpy.sin(psi)
            + 4 * c4 * numpy.cos(psi) ** 3 * numpy.sin(psi)
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


class DihedralSpline(BondSpline):
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

    def __init__(self, types, num_knots, mode="diff", name=None):
        super().__init__(types=types, num_knots=num_knots, mode=mode, name=name)

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

        if phi[0] != 0.0 and phi[-1] != numpy.pi:
            raise ValueError("The first and last knot must be at 0 and pi.")
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

        """
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

        """
        return super().force(type_=type_, x=phi)

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
            The bond distance(s) at which to evaluate the derivative.

        Returns
        -------
        float or numpy.ndarray
            The bond derivative evaluated at ``phi``. The return type is consistent
            with ``phi``.

        Raises
        ------
        ValueError
            If any value in ``phi`` is negative.
        TypeError
            If the parameter with respect to which to take the derivative
            is not a :class:`~relentless.variable.Variable`.

        """

        return super().derivative(type_=type_, var=var, x=phi)

    def _derivative(self, param, phi, **params):
        """Evaluate dihedral derivative with respect to a variable."""
        return super()._derivative(param=param, x=phi, **params)
