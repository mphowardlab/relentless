__all__ = ['Interpolator']

import numpy as np
import scipy.interpolate

class Interpolator:
    """Interpolating function.

    Interpolates through a function :math:`y(x)` on the domain
    :math:`a \le x \le b` using Akima splines. Outside this domain, `y` is
    extrapolated as a constant, i.e., :math:`y(x < a) = y(a)` and
    :math:`y(x > b) = y(b)`.

    Parameters
    ----------
    x : array_like
        1-d array of x coordinates that must be continually increasing.
    y : array_like
        1-d array of y coordinates.

    Raises
    ------
    ValueError
        If x is a scalar
    ValueError
        If x is not 1-dimensional
    ValueError
        If y is not the same shape as x
    ValueError
        If x is not strictly increasing

    Examples
    --------
    Interpolating the line :math:`y=2x`::

        f = Interpolator(x=(-1,0,1), y=(-2,0,2))

    Evaluating the function::

        >>> f(0.5)
        1.0
        >>> f([-0.5,0.5])
        (-1.0, 1.0)

    Evaluate the n-th derivative of the function::

        >>> f.derivative(x=0.5, n=1)
        2.0
        >>> f.derivative(x=[-2.5,-0.5,0.5,2.5], n=1)
        (0.0, 2.0, 2.0, 0.0)

    Extrapolation::

        >>> f(100)
        2.0

    """
    def __init__(self, x, y):
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        if x.shape[0] == 1:
            raise ValueError('x cannot be a scalar')
        if x.ndim > 1:
            raise ValueError('x must be 1-dimensional')
        if x.shape != y.shape:
            raise ValueError('x and y must be the same shape')
        if not np.all(x[1:] > x[:-1]):
            raise ValueError('x must be strictly increasing')
        self._domain = (x[0],x[-1])
        if x.shape[0] > 2:
            self._spline = scipy.interpolate.Akima1DInterpolator(x=x, y=y)
        else:
            self._spline = scipy.interpolate.InterpolatedUnivariateSpline(x=x, y=y, k=1)

    def __call__(self, x):
        """Evaluate the interpolating function.

        Parameters
        ----------
        x : float or array_like
            1-d array of x coordinates to evaluate.

        Returns
        -------
        result : float or numpy.ndarray
            Interpolated values having the same form as `x`.

        """
        scalar_x = np.isscalar(x)
        x = np.atleast_1d(x)
        result = np.zeros(len(x))

        # clamp lo
        lo = x < self.domain[0]
        result[lo] = self._spline(self.domain[0])

        # clamp hi
        hi = x > self.domain[1]
        result[hi] = self._spline(self.domain[1])

        # evaluate in between
        flags = np.logical_and(~lo,~hi)
        result[flags] = self._spline(x[flags])

        if scalar_x:
            result = result.item()

        return result

    def derivative(self, x, n):
        """Evaluate the n-th derivative of the interpolating function.

        Parameters
        ----------
        x : float or array_like
            1-d array of x coordinates to evaluate.
        n : int
            The order of the derivative to take.

        Returns
        -------
        result : float or numpy.ndarray
            Interpolated derivative values having the same form as `x`.

        Raises
        ------
        ValueError
            If n is not a positive integer.

        """
        if not isinstance(n, int) and n <= 0:
            raise ValueError('n must be a positive integer')
        scalar_x = np.isscalar(x)
        x = np.atleast_1d(x)
        result = np.zeros(len(x))

        # clamp lo
        lo = x < self.domain[0]
        result[lo] = 0

        # clamp hi
        hi = x > self.domain[1]
        result[hi] = 0

        # evaluate in between
        flags = np.logical_and(~lo,~hi)
        result[flags] = self._spline.derivative(n)(x[flags])

        if scalar_x:
            result = result.item()

        return result

    @property
    def domain(self):
        """tuple: The valid domain for interpolation."""
        return self._domain

