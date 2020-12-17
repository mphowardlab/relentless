__all__ = ['FixedKeyDict','PairMatrix']

class FixedKeyDict:
    """Dictionary with fixed keys.

    Parameters
    ----------
    keys : array_like
        List of keys to be fixed (a key must be a `str`).
    default
        Initial values to fill in the dictionary, defaults to `None`.

    Raises
    ------
    TypeError
        If keys does not consist only of strings

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
        d.update(A=0.5)  #equivalent statement
        >>> print(d)
        {'A':0.5, 'B':1.0}

    Single-key dictionary still needs `keys` as a tuple::

        FixedKeyDict(keys=('A',))

    """
    def __init__(self, keys, default=None):
        if not all(isinstance(k, str) for k in keys):
            raise TypeError('All keys must be strings')
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
        if key not in self.keys:
            raise KeyError('Key {} is not in dictionary.'.format(key))
        return key

    def __getitem__(self, key):
        key = self._check_key(key)
        return self._data[key]

    def __setitem__(self, key, value):
        key = self._check_key(key)
        self._data[key] = value

    def __iter__(self):
        return iter(self._data)

    def __next__(self):
        return next(self._data)

    def __str__(self):
        return str(self._data)

    def clear(self):
        """Clear entries in dict, resetting to default."""
        for i in self.keys:
            self._data[i] = self._default

    def update(self, *data, **values):
        """Partially reassigns key values.

        If both positional argument (data) and keyword arguments (values)
        are given as parameters, any keys in values will take precedence over data.

        Parameters
        ----------
        data : `dict`
            The keys and values to be updated/over-written, in a dictionary form.
        values : kwargs
            The keys and values to be updated/over-written.

        Raises
        ------
        TypeError
            If more than one positional argument is given.

        """
        if len(data) > 1:
            raise TypeError('More than one positional argument is given')
        elif len(data) == 1:
            for key in data[0]:
                self[key] = data[0][key]
        for key in values:
            self[key] = values[key]

    def todict(self):
        """Convert the fixed-key dictionary to a standard dictionary.

        Returns
        -------
        dict
            A copy of the data in the dictionary.

        """
        return dict(self._data)

    @property
    def keys(self):
        """tuple: All keys in the dictionary."""
        return self._keys

class PairMatrix:
    """Generic matrix of values per-pair.

    Defines a symmetric matrix of parameters corresponding to `(i,j)` pairs.
    The matrix is essentially a dictionary of dictionaries, keyed on `(i,j)`.
    There is an equivalent virtual entry for `(j,i)`. (The pairs that are
    actually saved have `j >= i`.) The dictionary associated with each pair
    can have any number of entries in it, although a common use case is to have
    the same parameter stored per-pair.

    The pairs in the matrix are frozen from the list of `types` specified when
    the object is constructed. It is an error to access pairs that cannot be
    formed from `types`.

    The pair matrix emulates a dictionary, and its pairs are iterable.

    Parameters
    ----------
    types : array_like
        List of types (A type must be a `str`).

    Raises
    ------
    ValueError
        If initialization occurs with empty types
    TypeError
        If types does not consist of only strings

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

    Single-type matrix still needs types as a tuple::

        PairMatrix(types=('A',))

    """
    def __init__(self, types):
        if len(types) == 0:
            raise ValueError('Cannot initialize with empty types')
        if not all(isinstance(t, str) for t in types):
            raise TypeError('All types must be strings')
        self.types = tuple(types)

        # flood data with type pairs
        self._data = {}
        for i in self.types:
            for j in self.types:
                if j >= i:
                    self._data[i,j] = {}

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
            raise KeyError('Coefficient matrix requires a pair of types.')

        if key[0] not in self.types:
            raise KeyError('Type {} is not in coefficient matrix.'.format(key[0]))
        elif key[1] not in self.types:
            raise KeyError('Type {} is not in coefficient matrix.'.format(key[1]))

        if key[1] >= key[0]:
            return key
        else:
            return (key[1],key[0])

    def __getitem__(self, key):
        """Get all coefficients for the `(i,j)` pair."""
        i,j = self._check_key(key)
        return self._data[i,j]

    def __setitem__(self, key, value):
        """Set coefficients for the `(i,j)` pair."""
        i,j = self._check_key(key)
        self._data[i,j] = value

    def __iter__(self):
        return iter(self._data)

    def __next__(self):
        return next(self._data)

    def __str__(self):
        return str(self._data)

    @property
    def pairs(self):
        """tuple: All unique pairs in the matrix."""
        return tuple(self._data.keys())
