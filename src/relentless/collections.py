"""
======================================
Collections (`relentless.collections`)
======================================

.. currentmodule:: relentless.collections

.. autosummary::
    :toctree: generated/

    DefaultDict
    FixedKeyDict
    PairMatrix

"""

import collections
import copy


class FixedKeyDict(collections.abc.MutableMapping):
    """Dictionary with fixed keys.

    Parameters
    ----------
    keys : array_like
        List of keys to be fixed.
    default : scalar
        Initial value to fill in the dictionary, defaults to ``None``.

    Examples
    --------
    Create a keyed dictionary::

        d = FixedKeyDict(keys=('A','B'))

    Default values::

        >>> print(d)
        {'A': None, 'B': None}

    Set default values::

        d = FixedKeyDict(keys=('A','B'), default=0.0)
        >>> print(d)
        {'A':0.0, 'B':0.0}

    Iterate as a dictionary::

        for k in d:
            d[k] = 1.0

    Access by key::

        >>> d['A']
        1.0
        >>> d['B']
        1.0

    Partially reassign/update values::

        d.update({'A':0.5})
        d.update(A=0.5)  # equivalent statement
        >>> print(d)
        {'A':0.5, 'B':1.0}

    Single-key dictionary still needs ``keys`` as a tuple::

        FixedKeyDict(keys=('A',))

    """

    def __init__(self, keys, default=None):
        self._keys = tuple(keys)
        self._data = {}
        self._default = default
        self.clear()

    def _check_key(self, key):
        """Check that a type is in the dictionary.

        Returns
        -------
        key
            The type that is keyed in the dictionary.

        Raises
        ------
        KeyError
            If the key is not in the dictionary.

        """
        if key not in self._keys:
            raise KeyError("Key {} is not in dictionary.".format(key))
        return key

    def __getitem__(self, key):
        key = self._check_key(key)
        return self._data[key]

    def __setitem__(self, key, value):
        key = self._check_key(key)
        self._data[key] = value

    def __delitem__(self, key):
        key = self._check_key(key)
        self._data[key] = copy.deepcopy(self.default)

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._data)

    def clear(self):
        """Clear entries in the dictionary, resetting to default."""
        for i in self._keys:
            self._data[i] = copy.deepcopy(self._default)


class PairMatrix(collections.abc.MutableMapping):
    """Generic matrix of values per-pair.

    Defines a symmetric matrix of parameters corresponding to ``(i,j)`` pairs.
    The matrix is essentially a dictionary of dictionaries, keyed on ``(i,j)``.
    There is an equivalent virtual entry for ``(j,i)``. (The pairs that are
    actually saved have ``j >= i``.) The dictionary associated with each pair
    can have any number of entries in it, although a common use case is to have
    the same parameter stored per-pair.

    The pairs in the matrix are frozen from the list of types specified when
    the object is constructed. It is an error to access pairs that cannot be
    formed from ``types``.

    The pair matrix emulates a dictionary, and its pairs are iterable.

    Parameters
    ----------
    types : array_like
        List of types (A type must be a :class:`str`).

    Raises
    ------
    ValueError
        If initialization occurs with empty ``types``.
    TypeError
        If ``types`` does not consist of only strings.

    Examples
    --------
    Create a pair matrix::

        m = PairMatrix(types=('A','B'))

    Set a pair matrix value::

        m['A','A']['energy'] = 1.0
        m['A','B']['energy'] = -1.0
        m['B','B']['energy'] = 1.0

    Get a pair matrix value::

        >>> m['A','A']['energy']
        1.0
        >>> m['A','B']['energy']
        -1.0
        >>> m['B','A']['energy']
        -1.0

    Iterate a pair matrix::

        for pair in m:
            m[pair]['mass'] = 1.0

    Multiple parameters are a dictionary::

        >>> m['A','B']
        {'energy': -1.0, 'mass': 1.0}

    Single-type matrix still needs ``types`` as a tuple::

        PairMatrix(types=('A',))

    """

    def __init__(self, keys, default=None):
        self._keys = tuple(keys)
        self._default = default
        self._data = {}
        self.clear()

    def _check_key(self, key):
        """Check that a pair key is valid.

        Returns
        -------
        tuple
            The `(i,j)` pair that is keyed in the dictionary.

        Raises
        ------
        KeyError
            If the key is not the right length or is not in the matrix.

        """
        if len(key) != 2:
            raise KeyError("Coefficient matrix requires a pair of types.")

        if key[0] not in self._keys:
            raise KeyError(f"Key {key[0]} is not in pair matrix.")
        elif key[1] not in self._keys:
            raise KeyError(f"Key {key[0]} is not in pair matrix.")

        if key[1] >= key[0]:
            return key
        else:
            return (key[1], key[0])

    def __getitem__(self, key):
        """Get all coefficients for the `(i,j)` pair."""
        key = self._check_key(key)
        return self._data[key]

    def __setitem__(self, key, value):
        """Set coefficients for the `(i,j)` pair."""
        key = self._check_key(key)
        self._data[key] = value

    def __delitem__(self, key):
        key = self._check_key(key)
        self._data[key] = {}

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def clear(self):
        """Clear entries in the dictionary, resetting to default."""
        for i in self._keys:
            for j in self._keys:
                if j >= i:
                    self._data[i, j] = copy.deepcopy(self._default)


class DefaultDict(collections.abc.MutableMapping):
    """Dictionary which supports a default value.

    Parameters
    ----------
    default : float
        The default value.

    """

    def __init__(self, default):
        self._data = {}
        self.default = default

    def __delitem__(self, key):
        del self._data[key]

    def __getitem__(self, key):
        """Get keyed item or default value if key is invalid."""
        try:
            return self._data[key]
        except KeyError:
            return self.default

    def __iter__(self):
        return iter(self._data)

    def __setitem__(self, key, value):
        """Set value of keyed item."""
        if key is None:
            raise KeyError("A DefaultDict key cannot be None.")
        self._data[key] = value

    def __len__(self):
        return len(self._data)

    @property
    def default(self):
        """float or dict: The default value."""
        return self._default

    @default.setter
    def default(self, value):
        self._default = value
