import numpy

from . import potential


class AngleParameters(potential.Parameters):
    """Parameters for a angle potential."""

    pass


class AnglePotential(potential.BondedPotential):
    r"""Abstract base class for an angle potential.

    The angle potential is defined by the angle between three bonded particles.
    The angle between particles :math:`i`, :math:`j`, and :math:`k` is defined as:

    .. math::

        \theta_{ijk} = \arccos\left(\frac{\mathbf{r}_{ij} \cdot
        \mathbf{r}_{jk}}{|\mathbf{r}_{ij}||\mathbf{r}_{jk}|}\right)

    where :math:`\mathbf{r}_{ij}` and :math:`\mathbf{r}_{jk}` are the vectors
    between particles :math:`i` and :math:`j`, and particles :math:`j` and
    :math:`k`, respectively.

    """

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
    | ``theta0``  | Minimum-energy angle :math:`\theta_0`.           |
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

        >>> u = relentless.potential.angle.HarmonicAngle(("A",))
        >>> u.coeff["A"].update({'k': 1000, 'theta0': 1.9})

    """

    def __init__(self, types, name=None):
        super().__init__(keys=types, params=("k", "theta0"), name=name)

    def energy(self, type_, theta):
        """Evaluate angle energy."""
        params = self.coeff.evaluate(type_)
        k = params["k"]
        theta0 = params["theta0"]

        theta, u, s = self._zeros(theta)

        u = 0.5 * k * (theta - theta0) ** 2

        if s:
            u = u.item()
        return u

    def force(self, type_, theta):
        """Evaluate angle force."""
        params = self.coeff.evaluate(type_)
        k = params["k"]
        theta0 = params["theta0"]

        theta, f, s = self._zeros(theta)

        f = -k * (theta - theta0)

        if s:
            f = f.item()
        return f

    def _derivative(self, param, theta, k, theta0):
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


class HarmonicCosineAngle(AnglePotential):
    r"""Harmonic cosine angle potential.

    .. math::

        u(\theta) = \frac{k}{2} (cos(\theta) - cos(\theta_0))^2

    where :math:`\theta` is the angle between three bonded particles. The parameters
    for each type are:

    +-------------+--------------------------------------------------+
    | Parameter   | Description                                      |
    +=============+==================================================+
    | ``k``       | Spring constant :math:`k`.                       |
    +-------------+--------------------------------------------------+
    | ``theta0``  | Minimum-energy angle :math:`\theta_0`.           |
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
    Cosine Squared Angle::

        >>> u = relentless.potential.angle.HarmonicCosineAngle(("A",))
        >>> u.coeff["A"].update({'k': 1000, 'theta0': 1})

    """

    def __init__(self, types, name=None):
        super().__init__(keys=types, params=("k", "theta0"), name=name)

    def energy(self, type_, theta):
        """Evaluate angle energy."""
        params = self.coeff.evaluate(type_)
        k = params["k"]
        theta0 = params["theta0"]

        theta, u, s = self._zeros(theta)

        u = 0.5 * k * (numpy.cos(theta) - numpy.cos(theta0)) ** 2

        if s:
            u = u.item()
        return u

    def force(self, type_, theta):
        """Evaluate angle force."""
        params = self.coeff.evaluate(type_)
        k = params["k"]
        theta0 = params["theta0"]

        theta, f, s = self._zeros(theta)

        f = k * (numpy.cos(theta) - numpy.cos(theta0)) * numpy.sin(theta)

        if s:
            f = f.item()
        return f

    def _derivative(self, param, theta, k, theta0):
        r"""Evaluate angle derivative with respect to a variable."""
        theta, d, s = self._zeros(theta)

        if param == "k":
            d = 0.5 * (numpy.cos(theta) - numpy.cos(theta0)) ** 2
        elif param == "theta0":
            d = k * (numpy.cos(theta) - numpy.cos(theta0)) * numpy.sin(theta0)
        else:
            raise ValueError("Unknown parameter")

        if s:
            d = d.item()
        return d


class CosineAngle(AnglePotential):
    r"""Cosine angle potential.

    .. math::

        u(\theta) = k (1 + cos(\theta))

    where :math:`\theta` is the angle between three bonded particles. The parameters
    for each type are:

    +-------------+--------------------------------------------------+
    | Parameter   | Description                                      |
    +=============+==================================================+
    | ``k``       | Spring constant :math:`k`.                       |
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
    Cosine Angle::

        >>> u = relentless.potential.angle.CosineAngle(("A",))
        >>> u.coeff["A"].update({'k': 1000})

    """

    def __init__(self, types, name=None):
        super().__init__(keys=types, params=("k",), name=name)

    def energy(self, type_, theta):
        """Evaluate angle energy."""
        params = self.coeff.evaluate(type_)
        k = params["k"]

        theta, u, s = self._zeros(theta)

        u = k * (1 + numpy.cos(theta))

        if s:
            u = u.item()
        return u

    def force(self, type_, theta):
        """Evaluate angle force."""
        params = self.coeff.evaluate(type_)
        k = params["k"]

        theta, f, s = self._zeros(theta)

        f = k * numpy.sin(theta)

        if s:
            f = f.item()
        return f

    def _derivative(self, param, theta, k):
        r"""Evaluate angle derivative with respect to a variable."""
        theta, d, s = self._zeros(theta)

        if param == "k":
            d = 1 + numpy.cos(theta)
        else:
            raise ValueError("Unknown parameter")

        if s:
            d = d.item()
        return d


class AngleSpline(potential.BondedSpline, AnglePotential):
    """Spline angle potential."""

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
