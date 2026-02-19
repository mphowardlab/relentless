import numpy

from . import potential


class BondParameters(potential.Parameters):
    """Parameters for a bond potential."""

    pass


class BondPotential(potential.BondedPotential):
    r"""Abstract base class for a bond potential."""

    def derivative(self, type_, var, r):
        r"""Evaluate bond derivative with respect to a variable.

        The derivative is evaluated using the :meth:`_derivative` function for all
        :math:`u_{0,\lambda}(r)`.

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
        r : float or list
            The bond distance(s) at which to evaluate the derivative.

        Returns
        -------
        float or numpy.ndarray
            The bond derivative evaluated at ``r``. The return type is consistent
            with ``r``.

        Raises
        ------
        ValueError
            If any value in ``r`` is negative.
        TypeError
            If the parameter with respect to which to take the derivative
            is not a :class:`~relentless.variable.Variable`.

        """
        return super().derivative(type_=type_, var=var, x=r)

    def _validate_coordinate(self, r):
        """Validate the bond length ``r`` is positive.

        Parameters
        ----------
        r : float or list
            The bond distance(s) to validate.

        Raises
        ------
        ValueError
            If any value in ``r`` is not positive.
        """
        if numpy.any(numpy.less(r, 0)):
            raise ValueError("Bond distance must be non-negative")


class HarmonicBond(BondPotential):
    r"""Harmonic bond potential.

    .. math::

        u(r) = \frac{k}{2} (r - r_0)^2

    where :math:`r` is the distance between two bonded particles. The parameters
    for each type are:

    +-------------+--------------------------------------------------+
    | Parameter   | Description                                      |
    +=============+==================================================+
    | ``k``       | Spring constant :math:`k`.                       |
    +-------------+--------------------------------------------------+
    | ``r0``      | Minimum-energy length :math:`r_0`.               |
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
    coeff : :class:`BondParameters`
        Parameters of the potential for each type.

    Examples
    --------
    Harmonic Bond::

        >>> u = relentless.potential.bond.Harmonic(("A",))
        >>> u.coeff["A"].update({'k': 1000, 'r0': 1})

    """

    def __init__(self, types, name=None):
        super().__init__(keys=types, params=("k", "r0"), name=name)

    def energy(self, type_, r):
        """Evaluate bond energy."""
        params = self.coeff.evaluate(type_)
        k = params["k"]
        r0 = params["r0"]

        r, u, s = self._zeros(r)

        self._validate_coordinate(r)

        u = 0.5 * k * (r - r0) ** 2

        if s:
            u = u.item()
        return u

    def force(self, type_, r):
        """Evaluate bond force."""
        params = self.coeff.evaluate(type_)
        k = params["k"]
        r0 = params["r0"]

        r, f, s = self._zeros(r)

        self._validate_coordinate(r)

        f = -k * (r - r0)

        if s:
            f = f.item()
        return f

    def _derivative(self, param, r, k, r0):
        r"""Evaluate bond derivative with respect to a variable."""
        r, d, s = self._zeros(r)

        if param == "k":
            d = (r - r0) ** 2 / 2
        elif param == "r0":
            d = -k * (r - r0)
        else:
            raise ValueError("Unknown parameter")

        if s:
            d = d.item()
        return d


class FENEWCA(BondPotential):
    r"""Finitely extensible nonlinear elastic (FENE) + Weeks-Chandler-Andersen
    (WCA) bond interaction.

    .. math::

        u(r) = - \frac{1}{2} k r_0^2 \ln \left[ 1 - \left( \frac{r}{r_0}
                \right)^2 \right] + u_{\rm{WCA}}(r)

    where

    .. math::
        u_{\rm{WCA}}(r)  =
        \begin{cases} 4 \varepsilon \left[ \left( \frac{\sigma}{r}
                                \right)^{12} - \left( \frac{\sigma}{r}
                                \right)^6 \right]  + \varepsilon
                               & r < 2^{1/6}\sigma\\
        0 & r \ge 2^{1/6}\sigma
        \end{cases}

    where :math:`r` is the distance between two bonded particles. The parameters
    for each type are:

    +-------------+--------------------------------------------------+
    | Parameter   | Description                                      |
    +=============+==================================================+
    | ``k``       | Spring constant :math:`k`.                       |
    +-------------+--------------------------------------------------+
    | ``r0``      | Minimum-energy length :math:`r_0`.               |
    +-------------+--------------------------------------------------+
    | ``epsilon`` | Interaction energy :math:`\varepsilon`.          |
    +-------------+--------------------------------------------------+
    | ``sigma``   | Interaction length :math:`\sigma`.               |
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
    coeff : :class:`BondParameters`
        Parameters of the potential for each type.

    Examples
    --------
    Kremer-Grest bond (no repulsion included)::

        >>> u = relentless.potential.bond.FENEWCA(("A"))
        >>> u.coeff["A"].update({'k': 30, 'r0': 1.5})

    Kremer-Grest bond (repulsion included)::

        >>> u = relentless.potential.bond.FENEWCA(("A"))
        >>> u.coeff["A"].update({'k': 30, 'r0': 1.5, "epsilon":1, "sigma": 1})
    """

    def __init__(self, types, name=None):
        super().__init__(keys=types, params=("k", "r0", "epsilon", "sigma"), name=name)

    def energy(self, type_, r):
        """Evaluate bond energy."""
        params = self.coeff.evaluate(type_)
        k = params["k"]
        r0 = params["r0"]
        epsilon = params["epsilon"]
        sigma = params["sigma"]

        # initialize arrays
        r, u_fene, s = self._zeros(r)

        self._validate_coordinate(r)

        u_wca = u_fene.copy()

        # set flags for FENE potential
        fene_flag = numpy.less(r, r0)

        # evaluate FENE potential
        u_fene[fene_flag] = -0.5 * k * r0**2 * numpy.log(1 - (r[fene_flag] / r0) ** 2)
        u_fene[~fene_flag] = numpy.inf

        # evaluate WCA potential
        nonzero_flags = ~numpy.isclose(r, 0)
        wca_flags = numpy.logical_and(nonzero_flags, r < 2 ** (1 / 6) * sigma)
        r6_inv = numpy.power(sigma / r[wca_flags], 6)
        u_wca[wca_flags] = 4.0 * epsilon * (r6_inv**2 - r6_inv) + epsilon
        u_wca[~nonzero_flags] = numpy.inf

        if s:
            u_fene = u_fene.item()
            u_wca = u_wca.item()

        return u_fene + u_wca

    def force(self, type_, r):
        """Evaluate bond force."""
        params = self.coeff.evaluate(type_)
        k = params["k"]
        r0 = params["r0"]
        epsilon = params["epsilon"]
        sigma = params["sigma"]

        # initialize arrays
        r, f_fene, s = self._zeros(r)

        self._validate_coordinate(r)
        f_wca = f_fene.copy()

        # set flags for FENE potential
        fene_flag = numpy.less(r, r0)

        # evaluate FENE potential
        f_fene[fene_flag] = -(k * r[fene_flag]) / (1 - (r[fene_flag] / r0) ** 2)
        f_fene[~fene_flag] = numpy.inf

        # evaluate WCA potential
        nonzero_flags = ~numpy.isclose(r, 0)
        wca_flags = numpy.logical_and(nonzero_flags, r < 2 ** (1 / 6) * sigma)
        rinv = 1.0 / r[wca_flags]
        r6_inv = numpy.power(sigma * rinv, 6)
        f_wca[wca_flags] = (48.0 * epsilon * rinv) * (r6_inv**2 - 0.5 * r6_inv)
        f_wca[~nonzero_flags] = numpy.inf

        if s:
            f_fene = f_fene.item()
            f_wca = f_wca.item()

        return f_fene + f_wca

    def _derivative(self, param, r, k, r0, epsilon, sigma):
        r"""Evaluate bond derivative with respect to a variable."""
        # initialize arrays
        r, d, s = self._zeros(r)
        # set flags for FENE potential
        fene_flag = numpy.less(r, r0)

        # set flags for WCA potential
        nonzero_flags = ~numpy.isclose(r, 0)
        wca_flags = numpy.logical_and(nonzero_flags, r < 2 ** (1 / 6) * sigma)

        # set r**6 for WCA potential
        r6_inv = numpy.power(sigma / r[wca_flags], 6)

        if param == "k":
            d[fene_flag] = 0.5 * r0**2 * numpy.log(1 - (r[fene_flag] / r0) ** 2)
            d[~fene_flag] = numpy.inf
        elif param == "r0":
            d[fene_flag] = (k * r[fene_flag] ** 2) / (
                (1 - (r[fene_flag] / r0) ** 2) * r0
            ) + k * r0 * numpy.log(1 - (r[fene_flag] / r0) ** 2)
            d[~fene_flag] = numpy.inf
        elif param == "epsilon":
            d[wca_flags] = 4 * (r6_inv**2 - r6_inv) + 1
            d[~nonzero_flags] = numpy.inf
        elif param == "sigma":
            d[wca_flags] = (48.0 * epsilon / sigma) * (r6_inv**2 - 0.5 * r6_inv)
            d[~nonzero_flags] = numpy.inf
        else:
            raise ValueError("Unknown parameter")
        if s:
            d = d.item()
        return d


class BondSpline(potential.BondedSpline, BondPotential):
    """Spline bond potentials.

    The bonded spline potential is defined by interpolation through a set of
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

        spline = relentless.potential.bond.BondSpline(types=("bondA",), num_knots=3)
        spline.from_array(("bondA"),[0,1,2],[4,2,0])

    However, the knot variables can be iterated over and manipulated directly::

        for r,k in spline.knots("bondA"):
            k.value = 1.0

    """

    _space_coord_name = "r"

    def from_array(self, types, r, u):
        r"""Set up the potential from knot points.

        Parameters
        ----------
        types : tuple[str]
            The type for which to set up the potential.
        r : list
            Position of each knot.
        u : list
            Potential energy of each knot.

        Raises
        ------
        ValueError
            If the number of ``r`` values is not the same as the number of knots.
        ValueError
            If the number of ``u`` values is not the same as the number of knots.

        """
        self._validate_coordinate(r)
        return super().from_array(types=types, x=r, u=u)

    def energy(self, type_, r):
        """Evaluate potential energy.

        Parameters
        ----------
        type_
            Type parametrizing the potential in :attr:`coeff<container>`.
        r : float or list
            Potential energy coordinate.

        Returns
        -------
        float or numpy.ndarray
            The pair energy evaluated at ``r``. The return type is consistent
            with ``r``.

        Raises
        ------
        ValueError
            If any value in ``r`` is negative.
        """
        self._validate_coordinate(r)
        return super().energy(type_=type_, x=r)

    def force(self, type_, r):
        """Evaluate force magnitude.

        The force is the (negative) magnitude of the ``x`` gradient.

        Parameters
        ----------
        type_
            Type parametrizing the potential in :attr:`coeff<container>`.
        x : float or list
            Potential energy coordinate.

        Returns
        -------
        float or numpy.ndarray
            The force evaluated at ``r``. The return type is consistent
            with ``r``.

        Raises
        ------
        ValueError
            If any value in ``r`` is negative.
        """
        self._validate_coordinate(r)

        return super().force(type_=type_, x=r)
