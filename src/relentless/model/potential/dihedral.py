import numpy

from relentless.model.potential.bond import BondSpline

from . import potential


class DihedralParameters(potential.Parameters):
    """Parameters for a dihedral potential."""

    pass


class DihedralPotential(potential.BondedPotential):
    r"""Abstract base class for an dihedral potential."""

    pass


class OPLSDihedral(DihedralPotential):
    r"""OPLS dihedral potential.

    .. math::

        u(\theta) = \frac{k}{2} (\theta - \theta_0)^2

    where :math:`\theta` is the dihedral between three bonded particles. The parameters
    for each type are:

    +-------------+--------------------------------------------------+
    | Parameter   | Description                                      |
    +=============+==================================================+
    | ``k``       | Spring constant :math:`k`.                       |
    +-------------+--------------------------------------------------+
    | ``theta0``  | Minimum-energy dihedral :math:`\thata_0`.           |
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

        >>> u = relentless.potential.dihedral.OPLSDihedral(("A"))
        >>> u.coeff["A"].update({'k': 1000, 'theta0': 1})

    """

    def __init__(self, types, name=None):
        super().__init__(keys=types, params=("k1", "k2", "k3", "k4"), name=name)

    def energy(self, types, phi):
        """Evaluate dihedral energy."""
        params = self.coeff.evaluate(types)
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

    def force(self, types, phi):
        """Evaluate dihedral force."""
        params = self.coeff.evaluate(types)
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

    def _derivative(self, param, phi, k1, k2, k3, k4, **params):
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

        u(\phi) = \frac{k}{2} (\phi - \phi_0)^2

    where :math:`\phi` is the dihedral between three bonded particles. The parameters
    for each type are:

    +-------------+--------------------------------------------------+
    | Parameter   | Description                                      |
    +=============+==================================================+
    | ``k``       | Spring constant :math:`k`.                       |
    +-------------+--------------------------------------------------+
    | ``phi0``  | Minimum-energy dihedral :math:`\thata_0`.           |
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

        >>> u = relentless.potential.dihedral.HarmonicDihedral(("A"))
        >>> u.coeff["A"].update({'k': 1000, 'phi0': 1})

    """

    def __init__(self, types, name=None):
        super().__init__(
            keys=types, params=("c0", "c1", "c2", "c3", "c4", "c5"), name=name
        )

    def energy(self, types, phi):
        """Evaluate dihedral energy."""
        params = self.coeff.evaluate(types)
        c0 = params["c0"]
        c1 = params["c1"]
        c2 = params["c2"]
        c3 = params["c3"]
        c4 = params["c4"]
        c5 = params["c5"]

        phi, u, s = self._zeros(phi)

        u = (
            c0
            + c1 * numpy.cos(phi)
            + c2 * numpy.cos(phi) ** 2
            + c3 * numpy.cos(phi) ** 3
            + c4 * numpy.cos(phi) ** 4
            + c5 * numpy.cos(phi) ** 5
        )

        if s:
            u = u.item()
        return u

    def force(self, types, phi):
        """Evaluate dihedral force."""
        params = self.coeff.evaluate(types)
        c1 = params["c1"]
        c2 = params["c2"]
        c3 = params["c3"]
        c4 = params["c4"]

        phi, f, s = self._zeros(phi)

        f = (
            c1 * numpy.sin(phi)
            + 2 * c2 * numpy.cos(phi) * numpy.sin(phi)
            + 3 * c3 * numpy.cos(phi) ** 2 * numpy.sin(phi)
            + 4 * c4 * numpy.cos(phi) ** 3 * numpy.sin(phi)
        )
        if s:
            f = f.item()
        return f

    def _derivative(self, param, phi, c0, c1, c2, c3, c4, c5, **params):
        r"""Evaluate dihedral derivative with respect to a variable."""
        phi, d, s = self._zeros(phi)

        if param == "c0":
            d = numpy.ones_like(phi)
        elif param == "c1":
            d = numpy.cos(phi)
        elif param == "c2":
            d = numpy.cos(phi) ** 2
        elif param == "c3":
            d = numpy.cos(phi) ** 3
        elif param == "c4":
            d = numpy.cos(phi) ** 4
        elif param == "c5":
            d = numpy.cos(phi) ** 5
        else:
            raise ValueError("Unknown parameter")

        print(s)
        if s:
            d = d.item()
        return d


class DihedralSpline(BondSpline):
    """Spline dihedral potential.

    The dihedral potential is defined by interpolation through a set of knot points.
    The interpolation scheme uses Akima splines.

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
            types=[dihedralA], num_knots=3
        )
        spline.from_array("dihedralA",[0,1,2],[4,2,0])

    However, the knot variables can be iterated over and manipulated directly::

        for r,k in spline.knots("dihedralA"):
            k.value = 1.0

    """

    def __init__(self, types, num_knots, mode="diff", name=None):
        super().__init__(types=types, num_knots=num_knots, mode=mode, name=name)

    def from_array(self, types, phi, u):
        return super().from_array(types=types, r=phi, u=u)

    def energy(self, types, phi):
        """Evaluate dihedral energy."""
        return super().energy(types=types, r=phi)

    def force(self, types, phi):
        """Evaluate dihedral force."""
        return super().force(types=types, r=phi)

    def _derivative(self, param, phi, **params):
        """Evaluate dihedral derivative with respect to a variable."""
        return super()._derivative(param=param, r=phi, **params)
