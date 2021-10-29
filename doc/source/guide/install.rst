============
Installation
============

The easiest way to get ``relentless`` is from PyPI using pip:

.. code:: bash

    pip install relentless

As different simulation software may be compatible (or not) with your computing
environment, **none** are installed by default. You should install the ones you
want to use. This is probably most easily managed with a virtual environment
in pip or conda.

Building from source
====================

If you want to install from source, you can also use pip (recommended)

.. code:: bash

    pip install .

or ``setuptools``

.. code:: bash

    python setup.py install

The required dependencies are pretty minimal:

- `Python <https://www.python.org>`_ (>= 3.6)
- `NumPy <https://numpy.org>`_
- `SciPy <https://www.scipy.org>`_
- `networkx <https://networkx.org>`_ (>= 2.4)

and can be installed from ``requirements.txt``.

Testing
-------

A suite of unit tests is provided with the source code and can be run
with ``unittest``:

.. code:: bash

    python -m unittest

Documentation
-------------

Documentation is hosted at `ReadTheDocs <https://relentless.readthedocs.io>`_,
but you can build it from source yourself:

.. code:: bash

    cd doc
    make html

You will need `sphinx <https://www.sphinx-doc.org>`_ (4.2.0) and the `Read
the Docs sphinx theme <https://sphinx-rtd-theme.readthedocs.io/en/stable>`_
(1.0.0), which can be installed using ``doc/requirements.txt``.
