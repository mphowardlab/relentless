import numpy

from relentless import math

from . import potential


class BondParameters(potential.Parameters):
    """Parameters for a bond potential."""

    pass


class BondPotential(potential.Potential):
    r"""Abstract base class for a bond potential."""

    pass


class Harmonic(BondPotential):
    r"""Harmonic bond potential.

    .. math::

        u(r) = \frac{k}{2} (r - r_0)^2

    where :math:`r` is the distance between two bonded particles. The parameters
    for each type are:

    +-------------+--------------------------------------------------+-----------+
    | Parameter   | Description                                      | Initial   |
    +=============+==================================================+===========+
    | ``k``       | potential constant                               |           |
    +-------------+--------------------------------------------------+-----------+
    | ``r0``      | rest length.                                     |           |
    +-------------+--------------------------------------------------+-----------+

    Parameters
    ----------
    type : str
        Type.
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

        >>> u = relentless.potential.bond.Harmonic("A")
        >>> u.coeff["A"].update({'k': 1.0, 'r0': 1.0})

    """

    def __init__(self, types, name=None):
        super().__init__(keys=types, params=("k", "r0"), name=name)

    def energy(self, type, r):
        """Evaluate bond energy."""
        k = self.coeff[type]["k"]
        r0 = self.coeff[type]["r0"]
        return k / 2 * (r - r0) ** 2

    def force(self, type, r):
        """Evaluate bond force."""
        k = self.coeff[type]["k"]
        r0 = self.coeff[type]["r0"]
        return -k * (r - r0)

    def derivative(self, type, var, r):
        r"""Evaluate bond derivative with respect to a variable."""
        k = self.coeff[type]["k"]
        r0 = self.coeff[type]["r0"]
        if var == "k":
            return (r - r0) ** 2 / 2
        elif var == "r0":
            return -k * (r - r0)
        else:
            raise ValueError("Unknown parameter")


class FENE(BondPotential):
    r"""The Finite Extensible Nonlinear Elastic (FENE) bond interaction.

    .. math::

        u(r) = -\frac{k r_0^2}{2} \ln \left(1 -
            \left(\frac{r - r_0}{r_0}\right)^2\right)

    where :math:`r` is the distance between two bonded particles. The parameters
    for each type are:

    +-------------+--------------------------------------------------+-----------+
    | Parameter   | Description                                      | Initial   |
    +=============+==================================================+===========+
    | ``k``       | attractive force strength.                       |           |
    +-------------+--------------------------------------------------+-----------+
    | ``r0``      |  size parameter.                                 |           |
    +-------------+--------------------------------------------------+-----------+

    Parameters
    ----------
    type : str
        Type.
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

        >>> u = relentless.potential.bond.FENE("A")
        >>> u.coeff["A"].update({'k': 1.0, 'r0': 1.0})
    """

    def __init__(self, types, name=None):
        super().__init__(keys=types, params=("k", "r0"), name=name)

    def energy(self, type, r):
        """Evaluate bond energy."""
        k = self.coeff[type]["k"]
        r0 = self.coeff[type]["r0"]
        return -0.5 * k * r0**2 * numpy.log(1 - (r - r0) ** 2 / r0**2)

    def force(self, type, r):
        """Evaluate bond force."""
        k = self.coeff[type]["k"]
        r0 = self.coeff[type]["r0"]
        return k * r0**2 / (r0**2 - (r - r0) ** 2)

    def derivative(self, type, var, r):
        r"""Evaluate bond derivative with respect to a variable."""
        k = self.coeff[type]["k"]
        r0 = self.coeff[type]["r0"]
        if var == "k":
            return -0.5 * r0**2 * numpy.log(1 - (r - r0) ** 2 / r0**2)
        elif var == "r0":
            return k * r0 * (r - r0) / (r0**2 - (r - r0) ** 2)
        else:
            raise ValueError("Unknown parameter")


class BondSpline(BondPotential):
    """Spline bond potential.

    The bond potential is defined by interpolation through a set of knot points.
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

        spline = relentless.potential.bond.BondSpline(types=[bondA], num_knots=3)
        spline.from_array("bondA",[0,1,2],[4,2,0])

    However, the knot variables can be iterated over and manipulated directly::

        for r,k in spline.knots("bondA"):
            k.value = 1.0

    """

    valid_modes = ("value", "diff")

    def __init__(self, types, num_knots, mode="diff", name=None):
        if isinstance(num_knots, int) and num_knots >= 2:
            self._num_knots = num_knots
        else:
            raise ValueError("Number of spline knots must be an integer >= 2.")

        if mode in self.valid_modes:
            self._mode = mode
        else:
            raise ValueError("Invalid mode, choose from: " + ",".join(self.valid_modes))

        params = []
        for i in range(self.num_knots):
            ri, ki = self.knot_params(i)
            params.append(ri)
            params.append(ki)
        super().__init__(keys=types, params=params, name=name)

    @classmethod
    def from_json(cls, data, name=None):
        u = super().from_json(data, name)
        # reset the knot values as variables since they were set as floats
        for type in u.coeff:
            for i, (r, k) in enumerate(u.knots(type)):
                u._set_knot(type, i, r, k)

        return u

    def from_array(self, type, r, u):
        r"""Set up the potential from knot points.

        Parameters
        ----------
        type : str
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
        # check that r and u have the right shape
        if len(r) != self.num_knots:
            raise ValueError("r must have the same length as the number of knots")
        if len(u) != self.num_knots:
            raise ValueError("u must have the same length as the number of knots")

        # convert to r,knot form given the mode
        rs = numpy.asarray(r, dtype=float)
        ks = numpy.asarray(u, dtype=float)
        if self.mode == "diff":
            # difference is next knot minus my knot,
            # with last knot fixed at its current value
            ks[:-1] -= ks[1:]

        # convert knot positions to differences
        drs = numpy.zeros_like(rs)
        drs[0] = rs[0]
        drs[1:] = rs[1:] - rs[:-1]

        for i in range(self.num_knots):
            self._set_knot(type, i, drs[i], ks[i])

    def to_json(self):
        data = super().to_json()
        data["num_knots"] = self.num_knots
        data["mode"] = self.mode
        return data

    def knot_params(self, i):
        r"""Get the parameter names for a given knot.

        Parameters
        ----------
        i : int
            Key for the knot variable.

        Returns
        -------
        str
            The parameter name of the :math:`r` value.
        str
            The parameter name of the knot value.

        Raises
        ------
        TypeError
            If the knot key is not an integer.

        """
        if not isinstance(i, int):
            raise TypeError("Knots are keyed by integers")
        return f"dr-{i}", f"{self.mode}-{i}"

    def _set_knot(self, type, i, dr, k):
        """Set the value of knot variables.

        The meaning of the value of the knot variable is defined by the ``mode``.
        This method is mostly meant to coerce the knot variable types.

        Parameters
        ----------
        type : str
            The type for which to set up the potential.
        i : int
            Index of the knot.
        r : float
            Relative position of each knot from previous one.
        u : float
            Value of the knot variable.

        """
        if i > 0 and dr <= 0:
            raise ValueError("Knot spacings must be positive")

        dri, ki = self.knot_params(i)
        self.coeff[type][dri] = dr
        self.coeff[type][ki] = k

    def energy(self, type, r):
        params = self.coeff.evaluate(type)
        r, u, s = self._zeros(r)
        u = self._interpolate(params)(r)
        if s:
            u = u.item()
        return u

    def force(self, type, r):
        params = self.coeff.evaluate(type)
        r, f, s = self._zeros(r)
        f = -self._interpolate(params).derivative(r, 1)
        if s:
            f = f.item()
        return f

    def derivative(self, type, r):
        params = self.coeff.evaluate(type)
        r, d, s = self._zeros(r)
        h = 0.001

        if "dr-" in params:
            f_low = self._interpolate(params)(r)
            knot_p = params[params]
            params[params] = knot_p + h
            f_high = self._interpolate(params)(r)
            params[params] = knot_p
            d = (f_high - f_low) / h
        elif self.mode + "-" in params:
            # perturb knot param value
            knot_p = params[params]
            params[params] = knot_p + h
            f_high = self._interpolate(params)(r)
            params[params] = knot_p - h
            f_low = self._interpolate(params)(r)
            params[params] = knot_p
            d = (f_high - f_low) / (2 * h)
        else:
            raise ValueError("Parameter cannot be differentiated")

        if s:
            d = d.item()
        return d

    def _interpolate(self, params):
        """Interpolate the knot points into a spline potential.

        Parameters
        ----------
        params : dict
            The knot parameters

        Returns
        -------
        :class:`~relentless.math.Interpolator`
            The interpolated spline potential.

        """
        dr = numpy.zeros(self.num_knots)
        u = numpy.zeros(self.num_knots)
        for i in range(self.num_knots):
            dri, ki = self.knot_params(i)
            dr[i] = params[dri]
            u[i] = params[ki]

        if numpy.any(dr[1:] <= 0):
            raise ValueError("Knot differences must be positive")

        # reconstruct r from differences
        r = numpy.cumsum(dr)

        # reconstruct the energies from differences, starting from the end
        if self.mode == "diff":
            u = numpy.flip(numpy.cumsum(numpy.flip(u)))

        return math.AkimaSpline(x=r, y=u)

    @property
    def num_knots(self):
        """int: Number of knots."""
        return self._num_knots

    @property
    def mode(self):
        """str: Spline construction mode."""
        return self._mode

    def knots(self, type):
        r"""Generator for knot points.

        Parameters
        ----------
        type : str
            The type for which to retrieve the knot points.

        Yields
        ------
        :class:`~relentless.variable.Variable`
            The next :math:`dr` variable in the parameters.
        :class:`~relentless.variable.Variable`
            The next knot variable in the parameters.

        """
        for i in range(self.num_knots):
            dri, ki = self.knot_params(i)
            yield self.coeff[type][dri], self.coeff[type][ki]
