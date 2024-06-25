"""
==================================
Math functions (`relentless.math`)
==================================

.. currentmodule:: relentless.math

.. autosummary::
    :toctree: generated/

    Interpolator
    AkimaSpline
    KeyedArray

"""

import numpy
import scipy.integrate
import scipy.interpolate

from .collections import FixedKeyDict


class Interpolator:
    r"""Interpolating function.

    Interpolates through a function :math:`y(x)` on the domain
    :math:`a \le x \le b`. Outside this domain, ``y`` is
    extrapolated as a constant, i.e., :math:`y(x < a) = y(a)` and
    :math:`y(x > b) = y(b)`\.

    Parameters
    ----------
    x : array_like
        1D array of x coordinates that must be continually increasing.
    y : array_like
        1D array of y coordinates.

    Raises
    ------
    ValueError
        If ``x`` is a scalar.
    ValueError
        If ``x`` is not 1-dimensional.
    ValueError
        If ``y`` is not the same shape as ``x``.
    ValueError
        If ``x`` is not strictly increasing.

    Examples
    --------
    Interpolating the line :math:`y=2x`::

        f = Interpolator(x=(-1,0,1), y=(-2,0,2))

    Evaluating the function::

        >>> f(0.5)
        1.0
        >>> f([-0.5,0.5])
        (-1.0, 1.0)

    Evaluate the :math:`n`\th derivative of the function::

        >>> f.derivative(x=0.5, n=1)
        2.0
        >>> f.derivative(x=[-2.5,-0.5,0.5,2.5], n=1)
        (0.0, 2.0, 2.0, 0.0)

    Extrapolation::

        >>> f(100)
        2.0

    """

    def __init__(self, x, y):
        x = numpy.atleast_1d(x)
        y = numpy.atleast_1d(y)
        if x.shape[0] == 1:
            raise ValueError("x cannot be a scalar")
        if x.ndim > 1:
            raise ValueError("x must be 1-dimensional")
        if x.shape != y.shape:
            raise ValueError("x and y must be the same shape")
        if not numpy.all(x[1:] > x[:-1]):
            raise ValueError("x must be strictly increasing")
        self._domain = (x[0], x[-1])
        self._table = numpy.column_stack((x, y))

        self._spline = None
        self._derivatives = {}

    def __call__(self, x):
        r"""Evaluate the interpolating function.

        Parameters
        ----------
        x : float or array_like
            1-d array of :math:`x` coordinates to evaluate.

        Returns
        -------
        result : float or numpy.ndarray
            Interpolated values having the same form as ``x``.

        """
        assert self._spline is not None

        scalar_x = numpy.isscalar(x)
        x = numpy.atleast_1d(x)
        result = numpy.zeros(len(x))

        # clamp lo
        lo = x < self.domain[0]
        result[lo] = self._spline(self.domain[0])

        # clamp hi
        hi = x > self.domain[1]
        result[hi] = self._spline(self.domain[1])

        # evaluate in between
        flags = numpy.logical_and(~lo, ~hi)
        result[flags] = self._spline(x[flags])

        if scalar_x:
            result = result.item()

        return result

    def derivative(self, x, n):
        r"""Evaluate the :math:`n`\th derivative of the interpolating function.

        Parameters
        ----------
        x : float or array_like
            1-d array of :math:`x` coordinates to evaluate.
        n : int
            The order of the derivative to take.

        Returns
        -------
        result : float or numpy.ndarray
            Interpolated derivative values having the same form as ``x``.


        """
        assert self._spline is not None

        if not isinstance(n, int) and n <= 0:
            raise ValueError("n must be a positive integer")
        if n not in self._derivatives:
            self._derivatives[n] = self._spline.derivative(n)

        scalar_x = numpy.isscalar(x)
        x = numpy.atleast_1d(x)
        result = numpy.zeros(len(x))

        # clamp lo
        lo = x < self.domain[0]
        result[lo] = 0

        # clamp hi
        hi = x > self.domain[1]
        result[hi] = 0

        # evaluate in between
        flags = numpy.logical_and(~lo, ~hi)
        result[flags] = self._derivatives[n](x[flags])

        if scalar_x:
            result = result.item()

        return result

    @property
    def domain(self):
        """tuple: The valid domain for interpolation."""
        return self._domain

    @property
    def table(self):
        """numpy.ndarray: The interpolated data."""
        return self._table


class AkimaSpline(Interpolator):
    """Interpolate using Akima splines.

    Parameters
    ----------
    x : array_like
        1D array of x coordinates that must be continually increasing.
    y : array_like
        1D array of y coordinates.


    """

    def __init__(self, x, y):
        super().__init__(x, y)

        if self._table.shape[0] > 2:
            self._spline = scipy.interpolate.Akima1DInterpolator(
                x=self._table[:, 0], y=self._table[:, 1]
            )
        else:
            self._spline = scipy.interpolate.InterpolatedUnivariateSpline(
                x=self._table[:, 0], y=self._table[:, 1], k=1
            )


class KeyedArray(FixedKeyDict):
    """Numerical array with fixed keys.

    Can be used to perform arithmetic operations between two arrays (element-wise)
    or between an array and a scalar, as well as vector algebraic operations
    (norm, dot product).

    Parameters
    ----------
    keys : array_like
        List of keys to be fixed.
    default : scalar
        Initial value to fill in the dictionary, defaults to ``None``.

    Examples
    --------
    Create a keyed array::

        k1 = KeyedArray(keys=('A','B'))
        k2 = KeyedArray(keys=('A','B'))

    Set values through update::

        k1.update({'A':2.0, 'B':3.0})
        k2.update({'A':3.0, 'B':4.0})

    Perform array-array arithmetic operations::

        >>> print(k1 + k2)
        {'A':5.0, 'B':7.0}
        >>> print(k1 - k2)
        {'A':-1.0, 'B':-1.0}
        >>> print(k1*k2)
        {'A':6.0, 'B':12.0}
        >>> print(k1/k2)
        {'A':0.6666666666666666, 'B':0.75}
        >>> print(k1**k2)
        {'A':8.0, 'B':81.0}

    Perform array-scalar arithmetic operations::

        >>> print(k1 + 3)
        {'A':5.0, 'B':6.0}
        >>> print(3 - k1)
        {'A':1.0, 'B':0.0}
        >>> print(3*k1)
        {'A':6.0, 'B':9.0}
        >>> print(k1/10)
        {'A':0.2, 'B':0.3}
        >>> print(k1**2)
        {'A':4.0, 'B':9.0}
        >>> print(-k1)
        {'A':-2.0, 'B':-3.0}

    Compute vector dot product::

        >>> print(k1.dot(k2))
        18.0

    Compute vector norm::

        >>> print(k2.norm())
        5.0

    """

    def __init__(self, keys, default=None):
        super().__init__(keys, default)

    def _assert_same_keys(self, val):
        if self.keys() != val.keys():
            raise KeyError(
                "Both KeyedArrays must have identical keys to"
                " perform mathematical operations."
            )

    def __add__(self, val):
        """Element-wise addition of two arrays, or of an array and a scalar."""
        k = KeyedArray(keys=self.keys())
        if isinstance(val, KeyedArray):
            self._assert_same_keys(val)
            k.update({x: self[x] + val[x] for x in self})
        elif numpy.isscalar(val):
            k.update({x: self[x] + val for x in self})
        else:
            raise TypeError("A KeyedArray can only add a scalar or a KeyedArray.")
        return k

    def __radd__(self, val):
        """Element-wise addition of a scalar and an array."""
        k = KeyedArray(keys=self.keys())
        if numpy.isscalar(val):
            k.update({x: val + self[x] for x in self})
        else:
            raise TypeError("A KeyedArray can only add a scalar or a KeyedArray.")
        return k

    def __iadd__(self, val):
        """In-place element-wise addition of two arrays, or of an array or scalar."""
        if isinstance(val, KeyedArray):
            self._assert_same_keys(val)
            for x in self:
                self[x] += val[x]
        elif numpy.isscalar(val):
            for x in self:
                self[x] += val
        else:
            raise TypeError("A KeyedArray can only add a scalar or a KeyedArray.")
        return self

    def __sub__(self, val):
        """Element-wise subtraction of two arrays, or of an array and a scalar."""
        k = KeyedArray(keys=self.keys())
        if isinstance(val, KeyedArray):
            self._assert_same_keys(val)
            k.update({x: self[x] - val[x] for x in self})
        elif numpy.isscalar(val):
            k.update({x: self[x] - val for x in self})
        else:
            raise TypeError("A KeyedArray can only subtract a scalar or a KeyedArray.")
        return k

    def __rsub__(self, val):
        """Element-wise subtraction of a scalar and an array."""
        k = KeyedArray(keys=self.keys())
        if numpy.isscalar(val):
            k.update({x: val - self[x] for x in self})
        else:
            raise TypeError("A KeyedArray can only subtract a scalar or a KeyedArray.")
        return k

    def __isub__(self, val):
        """In-place element-wise subtraction of two arrays,
        or of an array and a scalar.
        """
        if isinstance(val, KeyedArray):
            self._assert_same_keys(val)
            for x in self:
                self[x] -= val[x]
        elif numpy.isscalar(val):
            for x in self:
                self[x] -= val
        else:
            raise TypeError("A KeyedArray can only subtract a scalar or a KeyedArray.")
        return self

    def __mul__(self, val):
        """Element-wise multiplication of two arrays, or of an array and a scalar."""
        k = KeyedArray(keys=self.keys())
        if isinstance(val, KeyedArray):
            self._assert_same_keys(val)
            k.update({x: self[x] * val[x] for x in self})
        elif numpy.isscalar(val):
            k.update({x: self[x] * val for x in self})
        else:
            raise TypeError("A KeyedArray can only multiply a scalar or a KeyedArray.")
        return k

    def __rmul__(self, val):
        """Element-wise multiplication of a scalar by an array."""
        k = KeyedArray(keys=self.keys())
        if numpy.isscalar(val):
            k.update({x: val * self[x] for x in self})
        else:
            raise TypeError("A KeyedArray can only multiply a scalar or a KeyedArray.")
        return k

    def __imul__(self, val):
        """In-place element-wise multiplication of two arrays,
        or of an array by a scalar.
        """
        if isinstance(val, KeyedArray):
            self._assert_same_keys(val)
            for x in self:
                self[x] *= val[x]
        elif numpy.isscalar(val):
            for x in self:
                self[x] *= val
        else:
            raise TypeError("A KeyedArray can only multiply a scalar or a KeyedArray.")
        return self

    def __truediv__(self, val):
        """Element-wise division of two arrays, or of an array by a scalar."""
        k = KeyedArray(keys=self.keys())
        if isinstance(val, KeyedArray):
            self._assert_same_keys(val)
            k.update({x: self[x] / val[x] for x in self})
        elif numpy.isscalar(val):
            k.update({x: self[x] / val for x in self})
        else:
            raise TypeError("A KeyedArray can only divide a scalar or a KeyedArray.")
        return k

    def __rtruediv__(self, val):
        """Element-wise division of a scalar by an array."""
        k = KeyedArray(keys=self.keys())
        if numpy.isscalar(val):
            k.update({x: val / self[x] for x in self})
        else:
            raise TypeError("A KeyedArray can only divide a scalar or a KeyedArray.")
        return k

    def __itruediv__(self, val):
        """In-place element-wise division of two arrays, or of an array by a scalar."""
        if isinstance(val, KeyedArray):
            self._assert_same_keys(val)
            for x in self:
                self[x] /= val[x]
        elif numpy.isscalar(val):
            for x in self:
                self[x] /= val
        else:
            raise TypeError("A KeyedArray can only divide a scalar or a KeyedArray.")
        return self

    def __pow__(self, val):
        """Element-wise exponentiation of an array by a scalar or by an array."""
        k = KeyedArray(keys=self.keys())
        if isinstance(val, KeyedArray):
            self._assert_same_keys(val)
            k.update({x: self[x] ** val[x] for x in self})
        elif numpy.isscalar(val):
            k.update({x: self[x] ** val for x in self})
        else:
            raise TypeError(
                "A KeyedArray can only be exponentiated by a scalar or by a KeyedArray."
            )
        return k

    def __neg__(self):
        """Element-wise negation of an array."""
        k = KeyedArray(keys=self.keys())
        k.update({x: -self[x] for x in self})
        return k

    def norm(self):
        r"""Vector :math:`\ell^2`-norm.

        For a vector :math:`\mathbf{x}=\left[x_1,\ldots,x_n\right]`, the
        Euclidean 2-norm :math:`\lVert\mathbf{x}\rVert` is computed as:

        .. math::

            \lVert\mathbf{x}\rVert = \sqrt{\sum_{k=1}^{n} {x_k}^2}

        Returns
        -------
        float
            The vector norm.

        """
        return numpy.linalg.norm(list(self.values()))

    def dot(self, val):
        r"""Vector dot product.

        For two vectors :math:`\mathbf{x}=\left[x_1,\ldots,x_n\right]` and
        :math:`\mathbf{y}=\left[y_1,\ldots,y_n\right]`, the vector dot product
        :math:`\mathbf{x}\cdot\mathbf{y}` is computed as:

        .. math::

            \mathbf{x}\cdot\mathbf{y} = \sum_{k=1}^{n} {x_k y_k}

        Parameters
        ----------
        val : :class:`KeyedArray`
            One of the arrays used to compute the dot product.

        Returns
        -------
        floats
            The vector dot product.

        """
        self._assert_same_keys(val)
        return numpy.sum([self[x] * val[x] for x in self])


def _trapezoid(y, x):
    """Wrapper around SciPy trapezoidal integration.

    This function is a compatibility layer around different versions of SciPy,
    which changed the name of the trapezoidal integration method.

    Parameters
    ----------
    y : array_like
        Integrand.
    x : array_like
        Independent variable.

    Returns
    -------
    float
        The integral.

    """
    try:
        trapz = scipy.integrate.trapezoid
    except AttributeError:
        trapz = scipy.integrate.trapz

    return trapz(y, x=x)
