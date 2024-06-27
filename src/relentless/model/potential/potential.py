import abc
import inspect
import json
import re

import numpy

from relentless import collections, mpi
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
