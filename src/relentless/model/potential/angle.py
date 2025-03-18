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
    from particle *i* to particle *j* and from particle *j* to particle *k*,
    respectively. :math:`\theta` is bound between :math:`0` and :math:`\pi`.

    """

    def derivative(self, type_, var, theta):
        r"""Evaluate derivative with respect to a variable.

        The derivative is evaluated using the :meth:`_derivative` function for all
        :math:`u_{0,\lambda}(theta)`.

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
        theta : float or list
            The bond distance(s) at which to evaluate the derivative.

        Returns
        -------
        float or numpy.ndarray
            The bond derivative evaluated at ``theta``. The return type is consistent
            with ``theta``.

        Raises
        ------
        ValueError
            If any value in ``theta`` is negative.
        TypeError
            If the parameter with respect to which to take the derivative
            is not a :class:`~relentless.variable.Variable`.

        """

        return super().derivative(type_=type_, var=var, x=theta)

    def _validate_coordinate(self, theta):
        """Validate the angle ``theta`` is between 0 and pi."""
        if numpy.any(numpy.less(theta, 0)) or numpy.any(numpy.greater(theta, numpy.pi)):
            raise ValueError("Angle must be between 0 and pi.")
        return theta


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

        # Validate theta
        theta = self._validate_coordinate(theta)

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

        # Validate theta
        theta = self._validate_coordinate(theta)

        f = -k * (theta - theta0)

        if s:
            f = f.item()
        return f

    def _derivative(self, param, theta, k, theta0):
        r"""Evaluate angle derivative with respect to a variable."""
        theta, d, s = self._zeros(theta)

        # Validate theta
        theta = self._validate_coordinate(theta)

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

        u(\theta) = \frac{k}{2} (\cos \theta - \cos \theta_0)^2

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

        # Validate theta
        theta = self._validate_coordinate(theta)

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

        # Validate theta
        theta = self._validate_coordinate(theta)

        f = k * (numpy.cos(theta) - numpy.cos(theta0)) * numpy.sin(theta)

        if s:
            f = f.item()
        return f

    def _derivative(self, param, theta, k, theta0):
        r"""Evaluate angle derivative with respect to a variable."""
        theta, d, s = self._zeros(theta)

        # Validate theta
        theta = self._validate_coordinate(theta)

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

        u(\theta) = k (1 + \cos \theta)

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

        # Validate theta
        theta = self._validate_coordinate(theta)

        u = k * (1 + numpy.cos(theta))

        if s:
            u = u.item()
        return u

    def force(self, type_, theta):
        """Evaluate angle force."""
        params = self.coeff.evaluate(type_)
        k = params["k"]

        theta, f, s = self._zeros(theta)

        # Validate theta
        theta = self._validate_coordinate(theta)

        f = k * numpy.sin(theta)

        if s:
            f = f.item()
        return f

    def _derivative(self, param, theta, k):
        r"""Evaluate angle derivative with respect to a variable."""
        theta, d, s = self._zeros(theta)

        # Validate theta
        theta = self._validate_coordinate(theta)

        if param == "k":
            d = 1 + numpy.cos(theta)
        else:
            raise ValueError("Unknown parameter")

        if s:
            d = d.item()
        return d


class AngleSpline(potential.BondedSpline, AnglePotential):
    """Spline angle potentials.

    The angle spline potential is defined by interpolation through a set of
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

        spline = relentless.potential.angle.AngleSpline(types=("angleA",), num_knots=3)
        spline.from_array(("angleA"),[0,1,2],[4,2,0])

    However, the knot variables can be iterated over and manipulated directly::

        for r,k in spline.knots("angleA"):
            k.value = 1.0

    """

    _space_coord_name = "theta"

    def from_array(self, types, theta, u):
        r"""Set up the potential from knot points.

        Parameters
        ----------
        types : tuple[str]
            The type for which to set up the potential.
        theta : list
            Position of each knot.
        u : list
            Potential energy of each knot.

        Raises
        ------
        ValueError
            If the number of ``theta`` values is not the same as the number of knots.
        ValueError
            If the number of ``u`` values is not the same as the number of knots.

        """

        if theta[0] != 0.0 and theta[-1] != numpy.pi:
            raise ValueError("The first and last knot must be at 0 and pi.")
        return super().from_array(types=types, x=theta, u=u)

    def energy(self, type_, theta):
        """Evaluate potential energy.

        Parameters
        ----------
        type_
            Type parametrizing the potential in :attr:`coeff<container>`.
        theta : float or list
            Potential energy coordinate.

        Returns
        -------
        float or numpy.ndarray
            The pair energy evaluated at ``theta``. The return type is consistent
            with ``theta``.

        Raises
        ------
        ValueError
            If any value in ``theta`` is negative.
        """
        # Validate theta
        theta = self._validate_coordinate(theta)

        return super().energy(type_=type_, x=theta)

    def force(self, type_, theta):
        """Evaluate force magnitude.

        The force is the (negative) magnitude of the ``theta`` gradient.

        Parameters
        ----------
        type_
            Type parametrizing the potential in :attr:`coeff<container>`.
        theta : float or list
            Potential energy coordinate.

        Returns
        -------
        float or numpy.ndarray
            The force evaluated at ``theta``. The return type is consistent
            with ``theta``.

        Raises
        ------
        ValueError
            If any value in ``theta`` is negative.
        """
        # Validate theta
        theta = self._validate_coordinate(theta)

        return super().force(type_=type_, x=theta)

    def derivative(self, type_, var, theta):
        r"""Evaluate derivative with respect to a variable.

        The derivative is evaluated using the :meth:`_derivative` function for all
        :math:`u_{0,\lambda}(theta)`.

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
        theta : float or list
            The bond distance(s) at which to evaluate the derivative.

        Returns
        -------
        float or numpy.ndarray
            The bond derivative evaluated at ``theta``. The return type is consistent
            with ``theta``.

        Raises
        ------
        ValueError
            If any value in ``theta`` is negative.
        TypeError
            If the parameter with respect to which to take the derivative
            is not a :class:`~relentless.variable.Variable`.

        """
        # Validate theta
        theta = self._validate_coordinate(theta)

        return super().derivative(type_=type_, var=var, x=theta)

    def _derivative(self, param, theta, **params):
        """Evaluate angle derivative with respect to a variable."""
        return super()._derivative(param=param, x=theta, **params)
