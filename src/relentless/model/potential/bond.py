import numpy

from . import potential


class BondParameters(potential.Parameters):
    """Parameters for a bond potential."""

    def __init__(self, types, params):
        super().__init__(types, params)


class BondPotential(potential.Potential):
    r"""Abstract base class for a bond potential."""

    def __init__(self, types, params, name=None):
        super().__init__(types, params, name)

    def energy(self, type, r):
        """Evaluate bond energy."""
        pass

    def force(self, type, r):
        """Evaluate bond force."""
        pass

    def derivative(self, type, var, r):
        r"""Evaluate bond derivative with respect to a variable."""
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
        super().__init__(types=types, params=("k", "r0"), name=name)

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
        super().__init__(types=types, params=("k", "r0"), name=name)

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
