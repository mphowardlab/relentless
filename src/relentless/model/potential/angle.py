import numpy

from relentless.model.potential.bond import BondSpline

from . import potential


class AngleParameters(potential.Parameters):
    """Parameters for a angle potential."""

    pass


class AnglePotential(potential.BondedPotential):
    r"""Abstract base class for an angle potential."""

    pass


class HarmonicAngle(AnglePotential):
    r"""Harmonic angle potential.

    .. math::

        u(\theta) = \frac{k}{2} (\theta - \theta_0)^2

    where :math:`\theta` is the angle between three bonded particles. The parameters
    for each type are:

    +-------------+--------------------------------------------------+
    | Parameter   | Description                                      |
    +=============+==================================================+
    | ``k``       | Spring constant :math:`k`.                       |
    +-------------+--------------------------------------------------+
    | ``theta0``  | Minimum-energy angle :math:`\thata_0`.           |
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
    coeff : :class:`AngleParameters`
        Parameters of the potential for each type.

    Examples
    --------
    Harmonic Angle::

        >>> u = relentless.potential.angle.HarmonicAngle(("A"))
        >>> u.coeff["A"].update({'k': 1000, 'theta0': 1})

    """

    def __init__(self, types, name=None):
        super().__init__(keys=types, params=("k", "theta0"), name=name)

    def energy(self, types, theta):
        """Evaluate angle energy."""
        params = self.coeff.evaluate(types)
        k = params["k"]
        theta0 = params["theta0"]

        theta, u, s = self._zeros(theta)

        u = 0.5 * k * (theta - theta0) ** 2

        if s:
            u = u.item()
        return u

    def force(self, types, theta):
        """Evaluate angle force."""
        params = self.coeff.evaluate(types)
        k = params["k"]
        theta0 = params["theta0"]

        theta, f, s = self._zeros(theta)

        f = -k * (theta - theta0)

        if s:
            f = f.item()
        return f

    def _derivative(self, param, theta, k, theta0, **params):
        r"""Evaluate angle derivative with respect to a variable."""
        theta, d, s = self._zeros(theta)

        if param == "k":
            d = (theta - theta0) ** 2 / 2
        elif param == "theta0":
            d = -k * (theta - theta0)
        else:
            raise ValueError("Unknown parameter")

        if s:
            d = d.item()
        return d


class CosineSquaredAngle(AnglePotential):
    r"""Cosine squared angle potential.

    .. math::

        u(\theta) = \frac{k}{2} (\theta - \theta_0)^2

    where :math:`\theta` is the angle between three bonded particles. The parameters
    for each type are:

    +-------------+--------------------------------------------------+
    | Parameter   | Description                                      |
    +=============+==================================================+
    | ``k``       | Spring constant :math:`k`.                       |
    +-------------+--------------------------------------------------+
    | ``theta0``  | Minimum-energy angle :math:`\thata_0`.           |
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
    coeff : :class:`AngleParameters`
        Parameters of the potential for each type.

    Examples
    --------
    Harmonic Angle::

        >>> u = relentless.potential.angle.HarmonicAngle(("A"))
        >>> u.coeff["A"].update({'k': 1000, 'theta0': 1})

    """

    def __init__(self, types, name=None):
        super().__init__(keys=types, params=("k", "theta0"), name=name)

    def energy(self, types, theta):
        """Evaluate angle energy."""
        params = self.coeff.evaluate(types)
        k = params["k"]
        theta0 = params["theta0"]

        theta, u, s = self._zeros(theta)

        u = k * (numpy.cos(theta) - numpy.cos(theta0)) ** 2

        if s:
            u = u.item()
        return u

    def force(self, types, theta):
        """Evaluate angle force."""
        params = self.coeff.evaluate(types)
        k = params["k"]
        theta0 = params["theta0"]

        theta, f, s = self._zeros(theta)

        f = 2 * k * (numpy.cos(theta) - numpy.cos(theta0)) * numpy.sin(theta)

        if s:
            f = f.item()
        return f

    def _derivative(self, param, theta, k, theta0, **params):
        r"""Evaluate angle derivative with respect to a variable."""
        theta, d, s = self._zeros(theta)

        if param == "k":
            d = (numpy.cos(theta) - numpy.cos(theta0)) ** 2
        elif param == "theta0":
            d = 2 * k * (numpy.cos(theta) - numpy.cos(theta0)) * numpy.sin(theta0)
        else:
            raise ValueError("Unknown parameter")

        if s:
            d = d.item()
        return d


class AngleSpline(BondSpline):
    """Spline angle potential.

    The angle potential is defined by interpolation through a set of knot points.
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

        spline = relentless.potential.angle.AngleSpline(types=[angleA], num_knots=3)
        spline.from_array("angleA",[0,1,2],[4,2,0])

    However, the knot variables can be iterated over and manipulated directly::

        for r,k in spline.knots("angleA"):
            k.value = 1.0

    """

    def __init__(self, types, num_knots, mode="diff", name=None):
        super().__init__(types=types, num_knots=num_knots, mode=mode, name=name)

    def from_array(self, types, theta, u):
        return super().from_array(types=types, r=theta, u=u)

    def energy(self, types, theta):
        """Evaluate angle energy."""
        return super().energy(types=types, r=theta)

    def force(self, types, theta):
        """Evaluate angle force."""
        return super().force(types=types, r=theta)

    def _derivative(self, param, theta, **params):
        """Evaluate angle derivative with respect to a variable."""
        return super()._derivative(param=param, r=theta, **params)
