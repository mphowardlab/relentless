import abc

import numpy

from relentless import collections, math
from relentless.model import variable

from . import potential


class PairParameters(potential.Parameters):
    """Parameters for pairs of types.

    A pair is a tuple of two types, and each type is a :class:`str`. A named
    list of parameters can be set for each pair of types. The parameters for a
    pair are assumed to be symmetric, so the pair ``(i,j)`` is the same as the
    pair ``(j,i)``.

    The same parameters can also be set for each type. These values are not
    used directly in evaluating the per-pair coefficients, but they can be used
    in a :class:`~relentless.variable.DependentVariable` set for each pair, e.g.,
    as part of a mixing rule.

    Parameters
    ----------
    types : tuple[str]
        Types.
    params : tuple[str]
        Required parameters.

    Raises
    ------
    ValueError
        If ``params`` is empty.
    TypeError
        If ``params`` is not only strings.

    Examples
    --------
    Create a pair parameters for types ``A`` and ``B`` with parameters
    ``epsilon`` and ``sigma``::

        coeff = PairParameters(types=('A','B'), params=('epsilon','sigma'))

    The parameters can be accessed or iterated using ``params``::

        >>> print(coeff.params)
        ('epsilon','sigma')

    Set a parameter for a pair by accessing it directly::

        coeff['A','A']['epsilon'] = 2.0

    Parameters can also be assigned using ``update``, which works the same
    as :meth:`dict.update`::

        coeff['A','A'].update(sigma=2.5)
        >>> print(coeff['A','A'])
        {'epsilon':2.0, 'sigma':2.5}

    Assigning a :class:`dict` to a pair using the ``=`` operator resets any
    parameters that are not specified::

        coeff['A','A'] = {'sigma': 2.5}
        >>> print(coeff['A','A'])
        {'epsilon':None, 'sigma':2.5}

    Parameters can be a mix of constants and :class:`~relentless.variable.Variable`
    objects. To get the *value* of all parameters, use :meth:`evaluate`::

        >>> coeff['A','A']['epsilon'] = relentless.variable.IndependentVariable(1.0)
        >>> print(coeff.evaluate(('A','A')))
        {'epsilon':1.0, 'sigma':2.5}

    """

    def __init__(self, types, params):
        super().__init__(types, params)
        # override data container of parent to use pair matrix
        self._data = collections.PairMatrix(
            keys=types, default=collections.FixedKeyDict(keys=self.params)
        )


class PairPotential(potential.Potential):
    r"""Abstract base class for a pair potential.

    This class can be extended to evaluate the energy, force, and parameter
    derivatives of a pair potential with a given functional form.
    :meth:`_energy` specifies the potential energy :math:`u_0(r)`, :meth:`_force`
    specifies the force :math:`f_0(r) = -\partial u_0/\partial r`, and
    :meth:`_derivative` specifies the derivative
    :math:`u_{0,\lambda} = \partial u_0/\partial \lambda` with respect to
    parameter :math:`\lambda`.

    .. rubric:: Truncation and shifting

    The underlying pair potential can be truncated at a minimum distance
    :math:`r_{\rm min}` and maximum distance :math:`r_{\rm max}`. The truncation
    scheme is based on that used in molecular dynamics simulations, where the
    force is discontinuous and the pair potential is continuous.

    This makes the effective pair potential

    .. math::

        u(r) = \begin{cases}
            u_0(r_{\rm min}),& r < r_{\rm min} \\
            u_0(r),& r_{\rm min} \le r \le r_{\rm max} \\
            u_0(r_{\rm max}),& r > r_{\rm max}
            \end{cases},

    the effective pair force

    .. math::

        f(r) = \begin{cases}
            0,& r < r_{\rm min} \\
            f_0(r),& r_{\rm min} \le r \le r_{\rm max} \\
            0,& r > r_{\rm max}
            \end{cases},

    and the effective pair derivative

    .. math::

        u_\lambda(r) = \begin{cases}
            u_{0,\lambda}(r_{\rm min}),& r < r_{\rm min} \\
            u_{0,\lambda}(r),& r_{\rm min} \le r \le r_{\rm max} \\
            u_{0,\lambda}(r_{\rm max}),& r > r_{\rm max}
            \end{cases}.

    Two special cases are the derivative with respect to :math:`r_{\rm min}`

    .. math::

            u_{r_{\rm min}}(r) = \begin{cases}
            -f_0(r_{\rm min}),& r < r_{\rm min} \\
            0,& \textrm{otherwise}
            \end{cases}

    and with respect to :math:`r_{\rm max}`:

    .. math::

            u_{r_{\rm max}}(r) = \begin{cases}
            -f_0(r_{\rm max}),& r > r_{\rm max} \\
            0,& \textrm{otherwise}
            \end{cases}.

    The pair potential can also be shifted to zero at :math:`r_{\rm max}`.
    Shifting subtracts :math:`u_0(r_{\rm max})` from all pieces of :math:`u(r)` and
    :math:`-f_0(r_{\rm max})` from all parameter derivatives. The force is
    unaffected by shifting the pair potential.

    The parameters ``rmin``, ``rmax``, and ``shift`` will automatically be
    included in :attr:`coeff` with initial values of ``False``, which indicate
    that the potential should not be truncated or shifted.

    Parameters
    ----------
    types : tuple[str]
        Types.
    params : tuple[str]
        Required parameters.
    name : str
        Unique name of the potential. Defaults to ``__u[id]``, where ``id`` is the
        unique integer ID of the potential.

    Attributes
    ----------
    coeff : :class:`PairParameters`
        Parameters of the potential for each pair.

    """

    def __init__(self, types, params, name=None):
        # force in standard potential parameters if they are not explicitly set
        params = list(params)
        if "rmin" not in params:
            params.append("rmin")
        if "rmax" not in params:
            params.append("rmax")
        if "shift" not in params:
            params.append("shift")

        super().__init__(types, params, PairParameters, name)

        for p in self.coeff:
            self.coeff[p]["rmin"] = False
            self.coeff[p]["rmax"] = False
            self.coeff[p]["shift"] = False

    def energy(self, pair, r):
        """Evaluate pair energy.

        The energy is evaluated using the :meth:`_energy` function for
        :math:`u_0(r)`. The truncation and shifting scheme is applied.

        Parameters
        ----------
        pair : tuple[str]
            The pair for which to calculate the energy.
        r : float or list
            The pair distance(s) at which to evaluate the energy.

        Returns
        -------
        float or numpy.ndarray
            The pair energy evaluated at ``r``. The return type is consistent
            with ``r``.

        Raises
        ------
        ValueError
            If any value in ``r`` is negative.
        ValueError
            If the potential is shifted without setting ``rmax``.

        """
        params = self.coeff.evaluate(pair)
        r, u, scalar_r = self._zeros(r)
        if any(r < 0):
            raise ValueError("r cannot be negative")

        # evaluate at points below rmax (if set) first, including rmin cutoff (if set)
        flags = numpy.ones(r.shape[0], dtype=bool)
        if params["rmin"] is not False:
            range_ = r < params["rmin"]
            flags[range_] = False
            u[range_] = self._energy(params["rmin"], **params)
        if params["rmax"] is not False:
            flags[r > params["rmax"]] = False
        u[flags] = self._energy(r[flags], **params)

        # if rmax is set, truncate or shift depending on the mode
        if params["rmax"] is not False:
            # with shifting, move the whole potential up
            # otherwise, set energy to constant for any r beyond rmax
            if params["shift"]:
                u[r <= params["rmax"]] -= self._energy(params["rmax"], **params)
            else:
                u[r > params["rmax"]] = self._energy(params["rmax"], **params)
        elif params["shift"] is True:
            raise ValueError("Cannot shift potential without rmax")

        # coerce u back into shape of the input
        if scalar_r:
            u = u.item()
        return u

    def force(self, pair, r):
        """Evaluate pair force.

        The force is evaluated using the :meth:`_force` function for
        :math:`f_0(r)`. The truncation and shifting scheme is applied.

        Parameters
        ----------
        pair : tuple[str]
            The pair for which to calculate the force.
        r : float or list
            The pair distance(s) at which to evaluate the force.

        Returns
        -------
        float or numpy.ndarray
            The pair force evaluated at ``r``. The return type is consistent
            with ``r``.

        Raises
        ------
        ValueError
            If any value in ``r`` is negative.

        """
        params = self.coeff.evaluate(pair)
        r, f, scalar_r = self._zeros(r)
        if any(r < 0):
            raise ValueError("r cannot be negative")

        # only evaluate at points inside [rmin,rmax], if specified
        flags = numpy.ones(r.shape[0], dtype=bool)
        if params["rmin"] is not False:
            flags[r < params["rmin"]] = False
        if params["rmax"] is not False:
            flags[r > params["rmax"]] = False
        f[flags] = self._force(r[flags], **params)

        # coerce f back into shape of the input
        if scalar_r:
            f = f.item()
        return f

    def derivative(self, pair, var, r):
        r"""Evaluate pair derivative with respect to a variable.

        The derivative is evaluated using the :meth:`_derivative` function for all
        :math:`u_{0,\lambda}(r)`. The truncation and shifting scheme is applied.

        The derivative will be carried out with respect to ``var`` for all
        :class:`~relentless.variable.Variable` parameters. The appropriate chain
        rules are handled automatically. If the potential does not depend on
        ``var``, the derivative will be zero by definition.

        Parameters
        ----------
        pair : tuple[str]
            The pair for which to calculate the derivative.
        var : :class:`~relentless.variable.Variable`
            The variable with respect to which the derivative is calculated.
        r : float or list
            The pair distance(s) at which to evaluate the derivative.

        Returns
        -------
        float or numpy.ndarray
            The pair derivative evaluated at ``r``. The return type is consistent
            with ``r``.

        Raises
        ------
        ValueError
            If any value in ``r`` is negative.
        TypeError
            If the parameter with respect to which to take the derivative
            is not a :class:`~relentless.variable.Variable`.
        ValueError
            If the potential is shifted without setting ``rmax``.

        """
        params = self.coeff.evaluate(pair)
        r, deriv, scalar_r = self._zeros(r)
        if any(r < 0):
            raise ValueError("r cannot be negative")
        if not isinstance(var, variable.Variable):
            raise TypeError(
                "Parameter with respect to which to take the derivative"
                " must be a Variable."
            )

        flags = numpy.ones(r.shape[0], dtype=bool)

        for p in self.coeff.params:
            # skip shift parameter
            if p == "shift":
                continue

            # try to take chain rule w.r.t. variable first
            p_obj = self.coeff[pair][p]
            if isinstance(p_obj, variable.DependentVariable):
                dp_dvar = p_obj.derivative(var)
            elif isinstance(p_obj, variable.IndependentVariable) and var is p_obj:
                dp_dvar = 1.0
            else:
                dp_dvar = 0.0

            # skip when dp_dvar is exactly zero, since this does not contribute
            if dp_dvar == 0.0:
                continue

            # now take the parameter derivative
            if p == "rmin":
                # rmin deriv
                flags = r < params["rmin"]
                deriv[flags] += -self._force(params["rmin"], **params) * dp_dvar
            elif p == "rmax":
                # rmax deriv
                if params["shift"]:
                    flags = r <= params["rmax"]
                    deriv[flags] += self._force(params["rmax"], **params) * dp_dvar
                else:
                    flags = r > params["rmax"]
                    deriv[flags] += -self._force(params["rmax"], **params) * dp_dvar
            else:
                # regular parameter derivative
                below = numpy.zeros(r.shape[0], dtype=bool)
                if params["rmin"] is not False:
                    below = r < params["rmin"]
                    deriv[below] += (
                        self._derivative(p, params["rmin"], **params) * dp_dvar
                    )
                above = numpy.zeros(r.shape[0], dtype=bool)
                if params["rmax"] is not False:
                    above = r > params["rmax"]
                    deriv[above] += (
                        self._derivative(p, params["rmax"], **params) * dp_dvar
                    )
                elif params["shift"]:
                    raise ValueError("Cannot shift without setting rmax.")
                flags = numpy.logical_and(~below, ~above)
                deriv[flags] += self._derivative(p, r[flags], **params) * dp_dvar
                if params["shift"]:
                    deriv -= self._derivative(p, params["rmax"], **params) * dp_dvar

        # coerce derivative back into shape of the input
        if scalar_r:
            deriv = deriv.item()
        return deriv

    @abc.abstractmethod
    def _energy(self, r, **params):
        """Implementation of the energy function.

        This abstract method defines the interface for computing the energy of
        a pair interaction. ``**params`` will include all the parameters from
        :class:`PairParameters`.

        Parameters
        ----------
        r : float or list
            The pair distance(s) at which to evaluate the energy.
        **params : kwargs
            Named parameters of the potential.

        Returns
        -------
        float or numpy.ndarray
            The pair energy evaluated at ``r``. The return type is consistent
            with ``r``.

        """
        pass

    @abc.abstractmethod
    def _force(self, r, **params):
        """Implementation of the force function.

        This abstract method defines the interface for computing the force of
        a pair interaction. ``**params`` will include all the parameters from
        :class:`PairParameters`. The force should be consistent with the gradient
        of the :meth:`_energy`.

        Parameters
        ----------
        r : float or list
            The pair distance(s) at which to evaluate the force.
        **params : kwargs
            Named parameters of the potential.

        Returns
        -------
        float or numpy.ndarray
            The pair force evaluated at ``r``. The return type is consistent
            with ``r``.

        """
        pass

    @abc.abstractmethod
    def _derivative(self, param, r, **params):
        """Implementation of the parameter derivative function.

        This abstract method defines the interface for computing the parameter
        derivative of a pair interaction. ``**params`` will include all the
        parameters from :class:`PairParameters`. The derivative should be
        consistent with :meth:`_energy`.

        Parameters
        ----------
        param : str
            Name of the parameter.
        r : float or list
            The pair distance(s) at which to evaluate the derivative.
        **params : kwargs
            Named parameters of the potential.

        Returns
        -------
        float or numpy.ndarray
            The pair derivative evaluated at ``r``. The return type is consistent
            with ``r``.

        """
        pass


class Depletion(PairPotential):
    r"""Depletion pair potential.

    The Asakura--Oosawa pairwise attraction between spherical particles due to
    implicit depletion from an idealized polymer solution:

    .. math::

        u(r) = -\frac{\pi P}{12 r}
            \left[\frac{1}{2}(\sigma_i+\sigma_j)+\sigma_d-r\right]^2
            \left[r^2+r(\sigma_i+\sigma_j+2\sigma_d) \right.
            \left.-\frac{3}{4}(\sigma_i-\sigma_j)^2\right]

    where :math:`r` is the distance between two particles. The parameters for
    each :math:`(i,j)` pair are:

    +-------------+--------------------------------------------------+-----------+
    | Parameter   | Description                                      | Initial   |
    +=============+==================================================+===========+
    | ``P``       | Osmotic pressure :math:`P` of depletant.         |           |
    +-------------+--------------------------------------------------+-----------+
    | ``sigma_i`` | Diameter :math:`\sigma_i` of type :math:`i`.     |           |
    +-------------+--------------------------------------------------+-----------+
    | ``sigma_j`` | Diameter :math:`\sigma_j` of type :math:`j`.     |           |
    +-------------+--------------------------------------------------+-----------+
    | ``sigma_d`` | Diameter :math:`\sigma_d` of depletant.          |           |
    +-------------+--------------------------------------------------+-----------+
    | ``rmin``    | Minimum distance cutoff :math:`r_{\rm min}`.     | ``False`` |
    |             | Force is zero and energy is constant for         |           |
    |             | :math:`r < r_{\rm min}`. Ignored if ``False``.   |           |
    +-------------+--------------------------------------------------+-----------+
    | ``rmax``    | Maximum distance cutoff :math:`r_{\rm max}`.     | ``False`` |
    |             | Force is zero and energy is constant for         |           |
    |             | :math:`r > r_{\rm max}`. If ``False``, the       |           |
    |             | cutoff is automatically set to                   |           |
    |             | :math:`(\sigma_i+\sigma_j)/2+\sigma_d`.          |           |
    +-------------+--------------------------------------------------+-----------+
    | ``shift``   | If ``True``, shift potential to zero at ``rmax``.| ``False`` |
    +-------------+--------------------------------------------------+-----------+

    For most physical systems, it is advisable to set ``P`` and ``sigma_d`` to
    the same value for all pairs. It is also recommended to leave ``rmax=False``
    so that the potential is cutoff at the distance set by the diameters in the
    model.

    Parameters
    ----------
    types : tuple[str]
        Types.
    name : str
        Unique name of the potential. Defaults to ``__u[id]``, where ``id`` is the
        unique integer ID of the potential.

    Attributes
    ----------
    coeff : :class:`PairParameters`
        Parameters of the potential for each pair.

    Examples
    --------
    Depletion attraction::

        >>> u = relentless.potential.pair.Depletion(('A',))
        >>> u.coeff['A','A'].update({
            'P': 2.0, 'sigma_i': 1.0, 'sigma_j': 1.0, 'sigma_d': 0.1})

    """

    def __init__(self, types, name=None):
        super().__init__(
            types=types, params=("P", "sigma_i", "sigma_j", "sigma_d"), name=name
        )

    class Cutoff(variable.DependentVariable):
        r"""Physical cutoff for depletion potential.

        The depletion potential is usually cutoff based on the diameters of the
        particles and depletant:

        .. math::

            r_{\rm max} = \frac{1}{2}(\sigma_i+\sigma_j)+\sigma_d

        Parameters
        ----------
        sigma_i : int, float, or :class:`Variable`
            Diameter :math:`\sigma_i` of particle of type *i*.
        sigma_j : int/float or :class:`Variable`
            Diameter :math:`\sigma_j` of particle of type *j*.
        sigma_d : int/float or :class:`Variable`
            Diameter :math:`\sigma_d` of depletant.

        """

        def __init__(self, sigma_i, sigma_j, sigma_d):
            super().__init__(sigma_i=sigma_i, sigma_j=sigma_j, sigma_d=sigma_d)

        def compute(self, sigma_i, sigma_j, sigma_d):
            return 0.5 * (sigma_i + sigma_j) + sigma_d

        def compute_derivative(self, param, sigma_i, sigma_j, sigma_d):
            if param == "sigma_i":
                return 0.5
            elif param == "sigma_j":
                return 0.5
            elif param == "sigma_d":
                return 1.0
            else:
                raise ValueError("Unknown parameter")

    def energy(self, pair, r):
        # Override parent method to set rmax as cutoff
        if self.coeff[pair]["rmax"] is False:
            self.coeff[pair]["rmax"] = self.Cutoff(
                self.coeff[pair]["sigma_i"],
                self.coeff[pair]["sigma_j"],
                self.coeff[pair]["sigma_d"],
            )
        return super().energy(pair, r)

    def _energy(self, r, P, sigma_i, sigma_j, sigma_d, **params):
        if sigma_i <= 0 or sigma_j <= 0 or sigma_d <= 0:
            raise ValueError("sigma_i, sigma_j, and sigma_d must all be positive")
        r, u, s = self._zeros(r)

        p1 = (0.5 * (sigma_i + sigma_j) + sigma_d - r) ** 2
        p2 = (
            r**2
            + r * (sigma_i + sigma_j + 2.0 * sigma_d)
            - 0.75 * (sigma_i - sigma_j) ** 2
        )
        u = -(numpy.pi * P * p1 * p2) / (12.0 * r)

        if s:
            u = u.item()
        return u

    def force(self, pair, r):
        # Override parent method to set rmax as cutoff
        if self.coeff[pair]["rmax"] is False:
            self.coeff[pair]["rmax"] = self.Cutoff(
                self.coeff[pair]["sigma_i"],
                self.coeff[pair]["sigma_j"],
                self.coeff[pair]["sigma_d"],
            )
        return super().force(pair, r)

    def _force(self, r, P, sigma_i, sigma_j, sigma_d, **params):
        if sigma_i <= 0 or sigma_j <= 0 or sigma_d <= 0:
            raise ValueError("sigma_i, sigma_j, and sigma_d must all be positive")
        r, f, s = self._zeros(r)

        p1 = r**2 - 0.25 * (sigma_i - sigma_j) ** 2
        p2 = (0.5 * (sigma_i + sigma_j) + sigma_d) ** 2 - r**2
        f = -(numpy.pi * P * p1 * p2) / (4.0 * r**2)

        if s:
            f = f.item()
        return f

    def derivative(self, pair, var, r):
        # Override parent method to set rmax as cutoff
        if self.coeff[pair]["rmax"] is False:
            self.coeff[pair]["rmax"] = self.Cutoff(
                self.coeff[pair]["sigma_i"],
                self.coeff[pair]["sigma_j"],
                self.coeff[pair]["sigma_d"],
            )
        return super().derivative(pair, var, r)

    def _derivative(self, param, r, P, sigma_i, sigma_j, sigma_d, **params):
        if sigma_i <= 0 or sigma_j <= 0 or sigma_d <= 0:
            raise ValueError("sigma_i, sigma_j, and sigma_d must all be positive")
        r, d, s = self._zeros(r)

        if param == "P":
            p1 = (0.5 * (sigma_i + sigma_j) + sigma_d - r) ** 2
            p2 = (
                r**2
                + r * (sigma_i + sigma_j + 2.0 * sigma_d)
                - 0.75 * (sigma_i - sigma_j) ** 2
            )
            d = -(numpy.pi * p1 * p2) / (12.0 * r)
        elif param == "sigma_i":
            p1 = (0.5 * (sigma_i + sigma_j) + sigma_d - r) * (
                r**2
                + r * (sigma_i + sigma_j + 2.0 * sigma_d)
                - 0.75 * (sigma_i - sigma_j) ** 2
            )
            p2 = (r + 1.5 * (sigma_j - sigma_i)) * (
                0.5 * (sigma_i + sigma_j) + sigma_d - r
            ) ** 2
            d = -(numpy.pi * P * (p1 + p2)) / (12.0 * r)
        elif param == "sigma_j":
            p1 = (0.5 * (sigma_i + sigma_j) + sigma_d - r) * (
                r**2
                + r * (sigma_i + sigma_j + 2.0 * sigma_d)
                - 0.75 * (sigma_i - sigma_j) ** 2
            )
            p2 = (r + 1.5 * (sigma_i - sigma_j)) * (
                0.5 * (sigma_i + sigma_j) + sigma_d - r
            ) ** 2
            d = -(numpy.pi * P * (p1 + p2)) / (12.0 * r)
        elif param == "sigma_d":
            p1 = (sigma_i + sigma_j + 2.0 * sigma_d - 2.0 * r) * (
                r**2
                + r * (sigma_i + sigma_j + 2.0 * sigma_d)
                - 0.75 * (sigma_i - sigma_j) ** 2
            )
            p2 = 2.0 * r * (0.5 * (sigma_i + sigma_j) + sigma_d - r) ** 2
            d = -(numpy.pi * P * (p1 + p2)) / (12.0 * r)
        else:
            raise ValueError(
                "The depletion parameters are P, sigma_i, sigma_j, and sigma_d."
            )

        if s:
            d = d.item()
        return d


class LennardJones(PairPotential):
    r"""Lennard-Jones 12-6 pair potential.

    The classic molecular simulation potential:

    .. math::

        u(r) = 4 \varepsilon\left[\left(\frac{\sigma}{r}\right)^{12}
                - \left(\frac{\sigma}{r}\right)^6 \right]

    where :math:`r` is the distance between two particles. The parameters for
    each :math:`(i,j)` pair are:

    +-------------+-----------------------------------------------+-----------+
    | Parameter   | Description                                   | Initial   |
    +=============+===============================================+===========+
    | ``epsilon`` | Interaction energy :math:`\varepsilon`.       |           |
    +-------------+-----------------------------------------------+-----------+
    | ``sigma``   | Interaction length :math:`\sigma`.            |           |
    +-------------+-----------------------------------------------+-----------+
    | ``rmin``    | Minimum distance cutoff :math:`r_{\rm min}`.  | ``False`` |
    |             | Force is zero and energy is constant for      |           |
    |             | :math:`r < r_{\rm min}`. Ignored if ``False``.|           |
    +-------------+-----------------------------------------------+-----------+
    | ``rmax``    | Maximum distance cutoff :math:`r_{\rm max}`.  | ``False`` |
    |             | Force is zero and energy is constant for      |           |
    |             | :math:`r > r_{\rm max}`. Ignored if ``False``.|           |
    +-------------+-----------------------------------------------+-----------+
    | ``shift``   | If ``True``, shift potential to zero          | ``False`` |
    |             | at ``rmax``.                                  |           |
    +-------------+-----------------------------------------------+-----------+

    For example, setting :math:`r_{\rm max} = 2^{1/6}\sigma` and ``shift=True``
    will give the purely repulsive Weeks--Chandler--Anderson potential that is
    often used to model nearly hard spheres.

    Parameters
    ----------
    types : tuple[str]
        Types.
    name : str
        Unique name of the potential. Defaults to ``__u[id]``, where ``id`` is the
        unique integer ID of the potential.

    Attributes
    ----------
    coeff : :class:`PairParameters`
        Parameters of the potential for each pair.

    Examples
    --------
    Standard Lennard-Jones parameters::

        >>> u = relentless.potential.pair.LennardJones(('A',))
        >>> u.coeff['A','A'].update({'epsilon': 1.0, 'sigma': 1.0, 'rmax': 3.0})
        >>> u.energy(('A','A'), 1.0)
        0.0
        >>> u.force(('A','A'), 2.**(1./6.))
        0.0

    """

    def __init__(self, types, name=None):
        super().__init__(types=types, params=("epsilon", "sigma"), name=name)

    def _energy(self, r, epsilon, sigma, **params):
        if sigma < 0:
            raise ValueError("sigma must be positive")
        r, u, s = self._zeros(r)
        flags = ~numpy.isclose(r, 0)

        # evaluate potential
        r6_inv = numpy.power(sigma / r[flags], 6)
        u[flags] = 4.0 * epsilon * (r6_inv**2 - r6_inv)
        u[~flags] = numpy.inf

        if s:
            u = u.item()
        return u

    def _force(self, r, epsilon, sigma, **params):
        if sigma < 0:
            raise ValueError("sigma must be positive")
        r, f, s = self._zeros(r)
        flags = ~numpy.isclose(r, 0)

        # evaluate force
        rinv = 1.0 / r[flags]
        r6_inv = numpy.power(sigma * rinv, 6)
        f[flags] = (48.0 * epsilon * rinv) * (r6_inv**2 - 0.5 * r6_inv)
        f[~flags] = numpy.inf

        if s:
            f = f.item()
        return f

    def _derivative(self, param, r, epsilon, sigma, **params):
        if sigma < 0:
            raise ValueError("sigma must be positive")
        r, d, s = self._zeros(r)
        flags = ~numpy.isclose(r, 0)

        # evaluate derivative
        r6_inv = numpy.power(sigma / r[flags], 6)
        if param == "epsilon":
            d[flags] = 4 * (r6_inv**2 - r6_inv)
        elif param == "sigma":
            d[flags] = (48.0 * epsilon / sigma) * (r6_inv**2 - 0.5 * r6_inv)
        else:
            raise ValueError("The Lennard-Jones parameters are sigma and epsilon.")
        d[~flags] = numpy.inf

        if s:
            d = d.item()
        return d


class PairSpline(PairPotential):
    """Spline pair potential.

    The pair potential is defined by interpolation through a set of knot points.
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

        spline = relentless.potential.pair.Spline(types=('A',), num_knots=3)
        spline.from_array(('A','A'),[0,1,2],[4,2,0])

    However, the knot variables can be iterated over and manipulated directly::

        for r,k in spline.knots(('A','A')):
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
        super().__init__(types=types, params=params, name=name)

    @classmethod
    def from_json(cls, data, name=None):
        u = super().from_json(data, name)
        # reset the knot values as variables since they were set as floats
        for pair in u.coeff:
            for i, (r, k) in enumerate(u.knots(pair)):
                u._set_knot(pair, i, r, k)

        return u

    def from_array(self, pair, r, u):
        r"""Set up the potential from knot points.

        Each knot will be converted into two :class:`~relentless.variable.Variable`
        objects consistent with the storage ``mode``.

        Parameters
        ----------
        pair : tuple[str]
            The type pair ``(i,j)`` for which to set up the potential.
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
            self._set_knot(pair, i, drs[i], ks[i])

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

    def _set_knot(self, pair, i, dr, k):
        """Set the value of knot variables.

        The meaning of the value of the knot variable is defined by the ``mode``.
        This method is mostly meant to coerce the knot variable types.

        Parameters
        ----------
        pair : tuple[str]
            The type pair ``(i,j)`` for which to set up the potential.
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
        if isinstance(self.coeff[pair][dri], variable.IndependentVariable):
            self.coeff[pair][dri].value = dr
        else:
            self.coeff[pair][dri] = variable.IndependentVariable(dr)

        if isinstance(self.coeff[pair][ki], variable.IndependentVariable):
            self.coeff[pair][ki].value = k
        else:
            self.coeff[pair][ki] = variable.IndependentVariable(k)

    def _energy(self, r, **params):
        r, u, s = self._zeros(r)
        u = self._interpolate(params)(r)
        if s:
            u = u.item()
        return u

    def _force(self, r, **params):
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

    def knots(self, pair):
        r"""Generator for knot points.

        Parameters
        ----------
        pair : tuple[str]
            The type pair ``(i,j)`` for which to retrieve the knot points.

        Yields
        ------
        :class:`~relentless.variable.Variable`
            The next :math:`dr` variable in the parameters.
        :class:`~relentless.variable.Variable`
            The next knot variable in the parameters.

        """
        for i in range(self.num_knots):
            dri, ki = self.knot_params(i)
            yield self.coeff[pair][dri], self.coeff[pair][ki]

    @property
    def design_variables(self):
        """tuple: Designable variables of the spline."""
        dvars = []
        for pair in self.coeff:
            for i, (dr, k) in enumerate(self.knots(pair)):
                if i != self.num_knots - 1:
                    dvars.append(k)
        return tuple(dvars)


class Yukawa(PairPotential):
    r"""Yukawa pair potential.

    The classic pair potential for screened electrostatics:

    .. math::

        u(r) = \varepsilon \frac{e^{-\kappa r}}{r}

    where :math:`r` is the distance between two particles. The parameters for
    each :math:`(i,j)` pair are:

    +-------------+-----------------------------------------------+-----------+
    | Parameter   | Description                                   | Initial   |
    +=============+===============================================+===========+
    | ``epsilon`` | Prefactor :math:`\varepsilon` (dimensions:    |           |
    |             | energy x length).                             |           |
    +-------------+-----------------------------------------------+-----------+
    | ``kappa``   | Inverse screening length :math:`\kappa`.      |           |
    +-------------+-----------------------------------------------+-----------+
    | ``rmin``    | Minimum distance cutoff :math:`r_{\rm min}`.  | ``False`` |
    |             | Force is zero and energy is constant for      |           |
    |             | :math:`r < r_{\rm min}`. Ignored if ``False``.|           |
    +-------------+-----------------------------------------------+-----------+
    | ``rmax``    | Maximum distance cutoff :math:`r_{\rm max}`.  | ``False`` |
    |             | Force is zero and energy is constant for      |           |
    |             | :math:`r > r_{\rm max}`. Ignored if ``False``.|           |
    +-------------+-----------------------------------------------+-----------+
    | ``shift``   | If ``True``, shift potential to zero          | ``False`` |
    |             | at ``rmax``.                                  |           |
    +-------------+-----------------------------------------------+-----------+

    Parameters
    ----------
    types : tuple[str]
        Types.
    name : str
        Unique name of the potential. Defaults to ``__u[id]``, where ``id`` is the
        unique integer ID of the potential.

    Attributes
    ----------
    coeff : :class:`PairParameters`
        Parameters of the potential for each pair.

    Examples
    --------
    Nominal Yukawa parameters::

        >>> u = relentless.potential.pair.Yukawa(('A',))
        >>> u.coeff['A','A'].update({'epsilon': 100.0, 'kappa': 2.5})
        >>> u.energy(('A','A'), 2.0)
        0.33689734995427334

    """

    def __init__(self, types, name=None):
        super().__init__(types=types, params=("epsilon", "kappa"), name=name)

    def _energy(self, r, epsilon, kappa, **params):
        if kappa < 0:
            raise ValueError("kappa must be positive")
        r, u, s = self._zeros(r)
        flags = ~numpy.isclose(r, 0)

        # evaluate potential
        u[flags] = epsilon * numpy.exp(-kappa * r[flags]) / r[flags]
        u[~flags] = numpy.inf

        if s:
            u = u.item()
        return u

    def _force(self, r, epsilon, kappa, **params):
        if kappa < 0:
            raise ValueError("kappa must be positive")
        r, f, s = self._zeros(r)
        flags = ~numpy.isclose(r, 0)

        # evaluate force
        f[flags] = (
            epsilon
            * numpy.exp(-kappa * r[flags])
            * (1.0 + kappa * r[flags])
            / r[flags] ** 2
        )
        f[~flags] = numpy.inf

        if s:
            f = f.item()
        return f

    def _derivative(self, param, r, epsilon, kappa, **params):
        if kappa < 0:
            raise ValueError("kappa must be positive")
        r, d, s = self._zeros(r)
        flags = ~numpy.isclose(r, 0)

        # evaluate derivative
        if param == "epsilon":
            d[flags] = numpy.exp(-kappa * r[flags]) / r[flags]
            d[~flags] = numpy.inf
        elif param == "kappa":
            d = -epsilon * numpy.exp(-kappa * r)
        else:
            raise ValueError("The Yukawa parameters are kappa and epsilon.")

        if s:
            d = d.item()
        return d
