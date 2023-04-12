============
Installation
============

The easiest way to get ``relentless`` is from PyPI using ``pip``:

.. code:: bash

    pip install relentless

or from conda-forge using ``conda`` or ``mamba``:

.. code:: bash

    conda install -c conda-forge relentless

As different simulation software may be compatible (or not) with your computing
environment, **none** is installed with ``relentless``. You should install the
ones you want to use following the documentation in :mod:`relentless.simulate`.

To enable MPI support, you also need to install ``mpi4py``. This package can be
finicky on HPC systems and not everyone needs it, so it is not installed with
``relentless``.

Building from source
====================

If you want to build from source, you can also use ``pip``:

.. code:: bash

    pip install .

If you are developing new code, include the ``-e`` option for an editable build.
You should then also install the developer tools:

.. code:: bash

    pip install -r requirements-dev.txt -r doc/requirements.txt

A suite of unit tests is provided with the source code and can be run with
``unittest``:

.. code:: bash

    python -m unittest

You can build the documentation from source with:

.. code:: bash

    cd doc
    make html
