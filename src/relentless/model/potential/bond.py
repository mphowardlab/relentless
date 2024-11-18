import numpy

from relentless import math
from relentless.model import variable

from . import potential


class BondParameters(potential.Parameters):
    """Parameters for a bond potential."""

    pass


class BondPotential(potential.BondedPotential):
    r"""Abstract base class for a bond potential."""

    pass


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

        >>> u = relentless.potential.bond.Harmonic(("A"))
        >>> u.coeff["A"].update({'k': 1000, 'r0': 1})

    """

    def __init__(self, types, name=None):
        super().__init__(keys=types, params=("k", "r0"), name=name)

    def energy(self, types, r):
        """Evaluate bond energy."""
        params = self.coeff.evaluate(types)
        k = params["k"]
        r0 = params["r0"]

        r, u, s = self._zeros(r)

        u = 0.5 * k * (r - r0) ** 2

        if s:
            u = u.item()
        return u

    def force(self, types, r):
        """Evaluate bond force."""
        params = self.coeff.evaluate(types)
        k = params["k"]
        r0 = params["r0"]

        r, f, s = self._zeros(r)

        f = -k * (r - r0)

        if s:
            f = f.item()
        return f

    def _derivative(self, param, r, k, r0, **params):
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
                                \right)^{6} \right]  + \varepsilon
                               & r < 2^{\frac{1}{6}}\sigma\\
        0 & r \ge 2^{\frac{1}{6}}\sigma
        \end{cases}

    where :math:`r` is the distance between two bonded particles. The parameters
    for each type are:

    +-------------+--------------------------------------------------+-----------+
    | Parameter   | Description                                      | Initial   |
    +=============+==================================================+===========+
    | ``k``       | Spring constant :math:`k`.                       |           |
    +-------------+--------------------------------------------------+-----------+
    | ``r0``      | Minimum-energy length :math:`r_0`.               |           |
    +-------------+--------------------------------------------------+-----------+
    | ``epsilon`` | Interaction energy :math:`\varepsilon`.          |     0     |
    +-------------+--------------------------------------------------+-----------+
    | ``sigma``   | Interaction length :math:`\sigma`.               |     0     |
    +-------------+--------------------------------------------------+-----------+

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

    def energy(self, types, r):
        """Evaluate bond energy."""
        params = self.coeff.evaluate(types)
        k = params["k"]
        r0 = params["r0"]
        epsilon = params["epsilon"]
        sigma = params["sigma"]

        # initialize arrays
        r, FENE, s = self._zeros(r)
        WCA = FENE.copy()

        # set flags for FENE potential
        fene_flag = ~numpy.greater_equal(r, r0)

        # evaluate FENE potential
        FENE[fene_flag] = -0.5 * k * r0**2 * numpy.log(1 - (r[fene_flag] / r0) ** 2)
        FENE[~fene_flag] = numpy.inf

        # set flags for WCA potential
        zero_flags = ~numpy.isclose(r, 0)
        wca_flags = r < 2 ** (1 / 6) * sigma

        # evaluate WCA potential
        r6_inv = numpy.power(sigma / r[zero_flags], 6)
        WCA[zero_flags] = 4.0 * epsilon * (r6_inv**2 - r6_inv) + epsilon
        WCA[~zero_flags] = numpy.inf
        WCA[~wca_flags] = 0

        if s:
            FENE = FENE.item()
            WCA = WCA.item()

        return FENE + WCA

    def force(self, types, r):
        """Evaluate bond force."""
        params = self.coeff.evaluate(types)
        k = params["k"]
        r0 = params["r0"]
        epsilon = params["epsilon"]
        sigma = params["sigma"]

        # initialize arrays
        r, dFENE_dr, s = self._zeros(r)
        dWCA_dr = dFENE_dr.copy()

        # set flags for FENE potential
        fene_flag = ~numpy.greater_equal(r, r0)

        # evaluate FENE potential
        dFENE_dr[fene_flag] = (k * r0) / (1 - (r[fene_flag] / r0) ** 2)
        dFENE_dr[~fene_flag] = numpy.inf

        # set flags for WCA potential
        zero_flags = ~numpy.isclose(r, 0)
        wca_flags = r < 2 ** (1 / 6) * sigma

        # evaluate WCA potential
        rinv = 1.0 / r[zero_flags]
        r6_inv = numpy.power(sigma * rinv, 6)
        dWCA_dr[zero_flags] = (48.0 * epsilon * rinv) * (r6_inv**2 - 0.5 * r6_inv)
        dWCA_dr[~zero_flags] = numpy.inf
        dWCA_dr[~wca_flags] = 0

        if s:
            dFENE_dr = dFENE_dr.item()
            dWCA_dr = dWCA_dr.item()

        return dFENE_dr + dWCA_dr

    def _derivative(self, param, r, k, r0, epsilon, sigma, **params):
        r"""Evaluate bond derivative with respect to a variable."""
        # initialize arrays
        r, d, s = self._zeros(r)

        # set flags for FENE potential
        fene_flag = ~numpy.greater_equal(r, r0)

        # set flags for WCA potential
        zero_flags = ~numpy.isclose(r, 0)
        wca_flags = r < 2 ** (1 / 6) * sigma

        # set r**6 for WCA potential
        r6_inv = numpy.power(sigma / r[zero_flags], 6)

        if param == "k":
            d[fene_flag] = 0.5 * r0**2 * numpy.log(1 - (r[fene_flag] / r0) ** 2)
            d[~fene_flag] = numpy.inf
        elif param == "r0":
            d[fene_flag] = (k * r[fene_flag] ** 2) / (
                (1 - (r[fene_flag] / r0) ** 2) * r0
            ) + k * r0 * numpy.log(1 - (r[fene_flag] / r0) ** 2)
            d[~fene_flag] = numpy.inf
        elif param == "epsilon":
            d[zero_flags] = 4 * (r6_inv**2 - r6_inv) + 1
            d[~zero_flags] = numpy.inf
            d[~wca_flags] = 0
        elif param == "sigma":
            d[zero_flags] = (48.0 * epsilon / sigma) * (r6_inv**2 - 0.5 * r6_inv)
            d[~zero_flags] = numpy.inf
            d[~wca_flags] = 0
        else:
            raise ValueError("Unknown parameter")
        if s:
            d = d.item()
        return d


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
        # check that r and u have the right shape
        if len(r) != self.num_knots:
            raise ValueError("r must have the same length as the number of knots")
        if len(u) != self.num_knots:
            raise ValueError("u must have the same length as the number of knots")

        # convert to r,knot form given the mode
        rs = numpy.asarray(r, dtype=float)
        ks = numpy.asarray(u, dtype=float)
        if self.mode == "diff":
            # difference is next knot minus m y knot,
            # with last knot fixed at its current value
            ks[:-1] -= ks[1:]

        # convert knot positions to differences
        drs = numpy.zeros_like(rs)
        drs[0] = rs[0]
        drs[1:] = rs[1:] - rs[:-1]

        for i in range(self.num_knots):
            self._set_knot(types, i, drs[i], ks[i])

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

    def _set_knot(self, types, i, dr, k):
        """Set the value of knot variables.

        The meaning of the value of the knot variable is defined by the ``mode``.
        This method is mostly meant to coerce the knot variable types.

        Parameters
        ----------
        types : tuple[str]
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
        if isinstance(self.coeff[types][dri], variable.IndependentVariable):
            self.coeff[types][dri].value = dr
        else:
            self.coeff[types][dri] = variable.IndependentVariable(dr)

        if isinstance(self.coeff[types][ki], variable.IndependentVariable):
            self.coeff[types][ki].value = k
        else:
            self.coeff[types][ki] = variable.IndependentVariable(k)

    def energy(self, types, r):
        params = self.coeff.evaluate(types)
        r, u, s = self._zeros(r)
        u = self._interpolate(params)(r)
        if s:
            u = u.item()
        return u

    def force(self, types, r):
        params = self.coeff.evaluate(types)
        r, f, s = self._zeros(r)
        f = -self._interpolate(params).derivative(r, 1)
        if s:
            f = f.item()
        return f

    def _derivative(self, param, r, **params):
        r, d, s = self._zeros(r)
        h = 0.001

        if "dr-" in param:
            f_low = self._interpolate(params)(r)
            knot_p = params[param]
            params[param] = knot_p + h
            f_high = self._interpolate(params)(r)
            params[param] = knot_p
            d = (f_high - f_low) / h
        elif self.mode + "-" in param:
            # perturb knot param value
            knot_p = params[param]
            params[param] = knot_p + h
            f_high = self._interpolate(params)(r)
            params[param] = knot_p - h
            f_low = self._interpolate(params)(r)
            params[param] = knot_p
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

    def knots(self, types):
        r"""Generator for knot points.

        Parameters
        ----------
        types : tuple[str]
            The types for which to retrieve the knot points.

        Yields
        ------
        :class:`~relentless.variable.Variable`
            The next :math:`dr` variable in the parameters.
        :class:`~relentless.variable.Variable`
            The next knot variable in the parameters.

        """
        for i in range(self.num_knots):
            dri, ki = self.knot_params(i)
            yield self.coeff[types][dri], self.coeff[types][ki]

    @property
    def design_variables(self):
        """tuple: Designable variables of the spline."""
        dvars = []
        for types in self.coeff:
            for i, (dr, k) in enumerate(self.knots(types)):
                if i != self.num_knots - 1:
                    dvars.append(k)
        return tuple(dvars)
