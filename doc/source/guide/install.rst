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

    pip install relentless

or ``setuptools``

.. code:: bash

    python setup.py install

The required dependencies are pretty minimal:

- Python (>= 3.6)
- NumPy
- SciPy
- networkx (>= 2.4)

and can be installed using ``requirements.txt``.

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

You will need a compatible sphinx and the Read the Docs sphinx theme, which can
be installed from ``doc/requirements.txt``.
