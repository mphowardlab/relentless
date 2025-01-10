import abc
import inspect
import json
import re

import numpy

from relentless import collections, math, mpi
from relentless.model import variable


class Parameters:
    """Parameters for types.

    Each type is a :class:`str`. A named list of parameters can be set for type.

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

    """

    def __init__(self, types, params):
        if len(types) == 0:
            raise ValueError("Cannot initialize with empty types")
        if not all(isinstance(t, str) for t in types):
            raise TypeError("All types must be strings")
        self.types = tuple(types)

        if len(params) == 0:
            raise ValueError("Params cannot be initialized as empty")
        if not all(isinstance(p, str) for p in params):
            raise TypeError("All parameters must be strings")
        self.params = tuple(params)

        self._data = collections.FixedKeyDict(keys=self.types)
        for t in self.types:
            self._data[t] = collections.FixedKeyDict(keys=self.params)

    @classmethod
    def from_json(cls, data):
        params = cls(data["types"], data["params"])
        for key in params:
            params[key].update(data["values"][str(key)])
        return params

    def evaluate(self, key):
        """Evaluate parameters.

        Parameters
        ----------
        key : str
            Key for which the parameters are evaluated.

        Returns
        -------
        params : dict
            The evaluated parameters.

        """
        params = {}
        for p in self.params:
            if self[key][p] is not None:
                params[p] = variable.evaluate(self[key][p])
            else:
                raise ValueError("Parameter {} is not set for {}.".format(p, str(key)))
        return params

    def to_json(self):
        """Export parameters to a JSON-compatible dictionary.

        Returns
        -------
        dict
            Evaluated parameters.

        """
        data = {
            "types": self.types,
            "params": self.params,
            "values": {str(key): self.evaluate(key) for key in self},
        }
        return data

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        for p in value:
            if p not in self.params:
                raise KeyError("Only known parameters can be set.")

        self[key].clear()
        self[key].update(value)

    def __iter__(self):
        return iter(self._data)

    def __next__(self):
        return next(self._data)


class Potential(abc.ABC):
    """Abstract base class for interaction potential.

    A Potential defines the potential energy abstractly, which can be parametrized
    on a ``key`` (like a type) and that is a function of an arbitrary scalar
    coordinate ``x``. Concrete :meth:`energy`, :meth:`force`, and :meth:`derivative`
    methods must be implemented to define the potential energy (and its derivatives).

    Parameters
    ----------
    keys : list
        Keys for parametrizing the potential.
    params : list
        Parameters of the potential.
    container : object
        Container for storing coefficients. By default, :class:`Parameters` is used.
        The constructor of the ``container`` must accept two arguments: ``keys``
        and ``params``.
    name : str
        Unique name of the potential. Defaults to ``__u[id]``, where ``id`` is the
        unique integer ID of the potential.

    """

    count = 0
    names = set()
    _default_name_pattern = re.compile(r"__u\[[0-9]+\]")

    def __init__(self, keys, params, container=None, name=None):
        # set unique id and name
        self.id = Potential.count
        if name is None:
            name = "__u[{}]".format(self.id)
        if name in self.names:
            raise ValueError(f"Potential name {name} already used")
        else:
            self.names.add(name)
            self.name = name
        Potential.count += 1

        if container is None:
            container = Parameters
        self.coeff = container(keys, params)

    @classmethod
    def from_json(cls, data, name=None):
        """Create potential from JSON data.

        It is assumed that the data is compatible with the pair potential.

        Parameters
        ----------
        data : dict
            JSON data for potential.
        name : str or bool or None
            Name of the potential. If a `str`, ``name`` overrides the value in
            the JSON data. If ``True``, the name in the JSON data is always
            preserved. If ``False``, the name in the JSON data is always ignored,
            and a default name is created. If ``None``, the value in the JSON
            data is used if it is not taken and does not match the default name
            pattern; otherwise, a new default name is generated.

        """
        # build build constructor arguments from data and create object
        args = []
        kwargs = {}
        for arg, arg_info in inspect.signature(cls.__init__).parameters.items():
            # self does not need to be specified
            if arg == "self":
                continue

            # extract value of parameter from JSON data
            if arg == "types":
                x = data["coeff"]["types"]
            elif arg == "name":
                if isinstance(name, str):
                    # override
                    x = name
                elif name is True:
                    # force to preserve
                    x = data["name"]
                elif name is False:
                    # force to default
                    x = None
                else:
                    # default the name to None if it is taken or matches default pattern
                    x = data["name"]
                    if x in Potential.names or re.fullmatch(
                        cls._default_name_pattern, x
                    ):
                        x = None
            else:
                x = data[arg]

            # filter argument packs
            if arg_info.kind == arg_info.POSITIONAL_ONLY:
                args.append(x)
            elif (
                arg_info.kind == arg_info.POSITIONAL_OR_KEYWORD
                or arg_info.kind == arg_info.KEYWORD_ONLY
            ):
                kwargs[arg] = x
            else:
                raise NotImplementedError("Argument type not supported")

        u = cls(*args, **kwargs)

        # set coefficient values, do this here in case u wants some to be variables
        for key in u.coeff:
            for param in u.coeff[key]:
                data_value = data["coeff"]["values"][str(key)][param]
                if isinstance(u.coeff[key][param], variable.IndependentVariable):
                    u.coeff[key][param].value = data_value
                else:
                    u.coeff[key][param] = data_value

        return u

    @classmethod
    def from_file(cls, filename, name=None):
        """Create potential from a JSON file.

        It is assumed that the JSON file is compatible with the potential type.

        Parameters
        ----------
        filename : str
            JSON file to load.
        name : str or bool or None
            Name of the potential. If a `str`, ``name`` overrides the value in
            the file. If ``True``, the name in the file is always preserved. If
            ``False``, the name in the file is always ignored, and a default
            name is created. If ``None``, the value in the file is used if it is
            not taken and does not match the default name pattern; otherwise, a
            new default name is generated.

        """
        data = mpi.world.load_json(filename)
        return cls.from_json(data, name)

    @abc.abstractmethod
    def energy(self, key, x):
        """Evaluate potential energy.

        Parameters
        ----------
        key
            Key parametrizing the potential in :attr:`coeff<container>`.
        x : float or list
            Potential energy coordinate.

        Returns
        -------
        float or numpy.ndarray
            The pair energy evaluated at ``x``. The return type is consistent
            with ``x``.

        """
        pass

    @abc.abstractmethod
    def force(self, key, x):
        """Evaluate force magnitude.

        The force is the (negative) magnitude of the ``x`` gradient.

        Parameters
        ----------
        key
            Key parametrizing the potential in :attr:`coeff<container>`.
        x : float or list
            Potential energy coordinate.

        Returns
        -------
        float or numpy.ndarray
            The force evaluated at ``x``. The return type is consistent
            with ``x``.

        """
        pass

    @abc.abstractmethod
    def derivative(self, key, var, x):
        """Evaluate potential parameter derivative.

        Parameters
        ----------
        key
            Key parametrizing the potential in :attr:`coeff<container>`.
        var : :class:`~relentless.variable.Variable`
            The variable with respect to which the derivative is calculated.
        x : float or list
            Potential energy coordinate.

        Returns
        -------
        float or numpy.ndarray
            The potential parameter derivative evaluated at ``x``. The return
            type is consistent with ``x``.

        """
        pass

    @classmethod
    def _zeros(cls, x):
        """Force input to a 1-dimensional array and make matching array of zeros.

        Parameters
        ----------
        x : float or list
            One-dimensional array of coordinates.

        Returns
        -------
        numpy.ndarray
            ``x`` coerced into a NumPy array.
        numpy.ndarray
            Array of zeros the same shape as ``x``.
        bool
            ``True`` if ``x`` was originally a scalar.

        Raises
        ------
        TypeError
            If x is not a 1-dimensional array

        """
        s = numpy.isscalar(x)
        x = numpy.array(x, dtype=float, ndmin=1)
        if len(x.shape) != 1:
            raise TypeError("Potential coordinate must be 1D array.")
        return x, numpy.zeros_like(x), s

    def to_json(self):
        """Export potential to a JSON-compatible dictionary.

        The JSON dictionary will contain the ``id`` and ``name`` of the potential,
        along with the JSON representation of its coefficients.

        Returns
        -------
        dict
            Potential.

        """
        data = {
            "id": self.id,
            "name": self.name,
            "coeff": self.coeff.to_json(),
        }
        return data

    def save(self, filename):
        """Save the potential to file as JSON data.

        Parameters
        ----------
        filename : str
            The name of the file to which to save the data.

        """
        data = self.to_json()
        with open(filename, "w") as f:
            json.dump(data, f, sort_keys=True, indent=4)


class BondedPotential(Potential):
    r"""Abstract base class for bonded interaction potential.

    This class can be extended to evaluate the energy, force, and parameter
    derivatives of a bonded potential with a given functional form.
    :meth:`energy` specifies the potential energy :math:`u_0(r)`, :meth:`force`
    specifies the force :math:`f_0(r) = -\partial u_0/\partial r`, and
    :meth:`_derivative` specifies the derivative
    :math:`u_{0,\lambda} = \partial u_0/\partial \lambda` with respect to
    parameter :math:`\lambda`.
    """

    def derivative(self, type_, var, x):
        r"""Evaluate bond derivative with respect to a variable.

        The derivative is evaluated using the :meth:`_derivative` function for all
        :math:`u_{0,\lambda}(x)`.

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
        x : float or list
            The bond distance(s) at which to evaluate the derivative.

        Returns
        -------
        float or numpy.ndarray
            The bond derivative evaluated at ``x``. The return type is consistent
            with ``x``.

        Raises
        ------
        ValueError
            If any value in ``x`` is negative.
        TypeError
            If the parameter with respect to which to take the derivative
            is not a :class:`~relentless.variable.Variable`.

        """
        params = self.coeff.evaluate(type_)
        x, deriv, scalar_x = self._zeros(x)
        if any(x < 0):
            raise ValueError("x cannot be negative")
        if not isinstance(var, variable.Variable):
            raise TypeError(
                "Parameter with respect to which to take the derivative"
                " must be a Variable."
            )

        flags = numpy.ones(x.shape[0], dtype=bool)

        for p in self.coeff.params:
            # try to take chain rule w.r.t. variable first
            p_obj = self.coeff[type_][p]
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
            below = numpy.zeros(x.shape[0], dtype=bool)
            above = numpy.zeros(x.shape[0], dtype=bool)

            flags = numpy.logical_and(~below, ~above)
            deriv[flags] += self._derivative(p, x[flags], **params) * dp_dvar

        # coerce derivative back into shape of the input
        if scalar_x:
            deriv = deriv.item()
        return deriv

    @abc.abstractmethod
    def _derivative(self, param, x, **params):
        """Implementation of the parameter derivative function.

        This abstract method defines the interface for computing the parameter
        derivative of a bond interaction. ``**params`` will include all the
        parameters from :class:`PairParameters`. The derivative should be
        consistent with :meth:`_energy`.

        Parameters
        ----------
        param : str
            Name of the parameter.
        x : float or list
            The bond distance(s) at which to evaluate the derivative.
        **params : kwargs
            Named parameters of the potential.

        Returns
        -------
        float or numpy.ndarray
            The bond derivative evaluated at ``x``. The return type is consistent
            with ``x``.

        """
        pass


class BondedSpline(BondedPotential):
    """Base class for bonded spline potentials.

    The bonded spline potential is defined by interpolation through a set of
    knot points. The interpolation scheme uses Akima splines.

    This class should not be instantiated directly. Instead, use the appropriate
    spline type, i.e., :class:`~relentless.model.potential.BondSpline` or
    :class:`~relentless.model.potential.AngleSpline`.

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

    """

    valid_modes = ("value", "diff")
    _space_coord_name = "x"

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
            xi, ki = self.knot_params(i)
            params.append(xi)
            params.append(ki)
        super().__init__(keys=types, params=params, name=name)

    @classmethod
    def from_json(cls, data, name=None):
        u = super().from_json(data, name)
        # reset the knot values as variables since they were set as floats
        for type in u.coeff:
            for i, (x, k) in enumerate(u.knots(type)):
                u._set_knot(type, i, x, k)

        return u

    def from_array(self, types, x, u):
        r"""Set up the potential from knot points.

        Parameters
        ----------
        types : tuple[str]
            The type for which to set up the potential.
        x : list
            Position of each knot.
        u : list
            Potential energy of each knot.

        Raises
        ------
        ValueError
            If the number of ``x`` values is not the same as the number of knots.
        ValueError
            If the number of ``u`` values is not the same as the number of knots.

        """
        # check that r and u have the right shape
        if len(x) != self.num_knots:
            raise ValueError("x must have the same length as the number of knots")
        if len(u) != self.num_knots:
            raise ValueError("u must have the same length as the number of knots")

        # convert to r,knot form given the mode
        xs = numpy.asarray(x, dtype=float)
        ks = numpy.asarray(u, dtype=float)
        if self.mode == "diff":
            # difference is next knot minus m y knot,
            # with last knot fixed at its current value
            ks[:-1] -= ks[1:]

        # convert knot positions to differences
        dxs = numpy.zeros_like(xs)
        dxs[0] = xs[0]
        dxs[1:] = xs[1:] - xs[:-1]

        for i in range(self.num_knots):
            self._set_knot(types, i, dxs[i], ks[i])

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
            The parameter name of the :math:`x` value.
        str
            The parameter name of the knot value.

        Raises
        ------
        TypeError
            If the knot key is not an integer.

        """
        if not isinstance(i, int):
            raise TypeError("Knots are keyed by integers")
        return f"d{self._space_coord_name}-{i}", f"{self.mode}-{i}"

    def _set_knot(self, types, i, dx, k):
        """Set the value of knot variables.

        The meaning of the value of the knot variable is defined by the ``mode``.
        This method is mostly meant to coerce the knot variable types.

        Parameters
        ----------
        types : tuple[str]
            The type for which to set up the potential.
        i : int
            Index of the knot.
        x : float
            Relative position of each knot from previous one.
        u : float
            Value of the knot variable.

        """
        if i > 0 and dx <= 0:
            raise ValueError("Knot spacings must be positive")

        dxi, ki = self.knot_params(i)
        if isinstance(self.coeff[types][dxi], variable.IndependentVariable):
            self.coeff[types][dxi].value = dx
        else:
            self.coeff[types][dxi] = variable.IndependentVariable(dx)

        if isinstance(self.coeff[types][ki], variable.IndependentVariable):
            self.coeff[types][ki].value = k
        else:
            self.coeff[types][ki] = variable.IndependentVariable(k)

    def energy(self, type_, x):
        """Evaluate potential energy.

        Parameters
        ----------
        type_
            Type parametrizing the potential in :attr:`coeff<container>`.
        x : float or list
            Potential energy coordinate.

        Returns
        -------
        float or numpy.ndarray
            The pair energy evaluated at ``x``. The return type is consistent
            with ``x``.

        """
        params = self.coeff.evaluate(type_)
        x, u, s = self._zeros(x)
        u = self._interpolate(params)(x)
        if s:
            u = u.item()
        return u

    def force(self, type_, x):
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
            The force evaluated at ``x``. The return type is consistent
            with ``x``.

        """
        params = self.coeff.evaluate(type_)
        x, f, s = self._zeros(x)
        f = -self._interpolate(params).derivative(x, 1)
        if s:
            f = f.item()
        return f

    def _derivative(self, param, x, **params):
        x, d, s = self._zeros(x)
        h = 0.001

        if f"d{self._space_coord_name}-" in param:
            f_low = self._interpolate(params)(x)
            knot_p = params[param]
            params[param] = knot_p + h
            f_high = self._interpolate(params)(x)
            params[param] = knot_p
            d = (f_high - f_low) / h
        elif self.mode + "-" in param:
            # perturb knot param value
            knot_p = params[param]
            params[param] = knot_p + h
            f_high = self._interpolate(params)(x)
            params[param] = knot_p - h
            f_low = self._interpolate(params)(x)
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
        dx = numpy.zeros(self.num_knots)
        u = numpy.zeros(self.num_knots)
        for i in range(self.num_knots):
            dxi, ki = self.knot_params(i)
            dx[i] = params[dxi]
            u[i] = params[ki]

        if numpy.any(dx[1:] <= 0):
            raise ValueError("Knot differences must be positive")

        # reconstruct r from differences
        x = numpy.cumsum(dx)

        # reconstruct the energies from differences, starting from the end
        if self.mode == "diff":
            u = numpy.flip(numpy.cumsum(numpy.flip(u)))

        return math.AkimaSpline(x=x, y=u)

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
            dxi, ki = self.knot_params(i)
            yield self.coeff[types][dxi], self.coeff[types][ki]

    @property
    def design_variables(self):
        """tuple: Designable variables of the spline."""
        dvars = []
        for types in self.coeff:
            for i, (dx, k) in enumerate(self.knots(types)):
                if i != self.num_knots - 1:
                    dvars.append(k)
        return tuple(dvars)
