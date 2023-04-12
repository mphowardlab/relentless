==========
relentless
==========

|ReadTheDocs|

Overview
========

``relentless`` is a Python package for executing molecular simulations in larger
computational workflows. The simulation protocol is specified as a reproducible,
human-readable recipe that is run natively in popular engines such as `LAMMPS`_
and `HOOMD-blue`_. ``relentless`` also has robust features for optimization
with simulations, including independent and dependent variables with automatic
differentiation, objective functions of simulations, and optimization methods.
All features are readily extensible through a clean hierarchy of abstract objects,
enabling you to quickly use your own simulation code or optimization objective.
Our initial goal was to enable optimization of the ``rel``\ ative ``ent``\ ropy
of structural coarse-graining and materials design: with ``less`` code.


Resources
=========

- `Documentation <https://relentless.readthedocs.io>`_:
  Installation, examples, and commands.
- `Source code <https://github.com/mphowardlab/relentless>`_:
  Download or contribute on GitHub.
- `Issue tracker <https://github.com/mphowardlab/relentless/issues>`_:
  Report issues or request features.

Installation
============

Install ``relentless`` from PyPI using ``pip``:

.. code:: bash

    pip install relentless

or from conda-forge using ``conda`` or ``mamba``

.. code:: bash

    conda install -c conda-forge relentless

Contributing
============

Contributions are welcomed and appreciated! Fork and create a pull request on
`GitHub <https://github.com/mphowardlab/relentless>`_. We value the input and
experiences all users and contributors bring to ``relentless``.

.. _HOOMD-blue: https://hoomd-blue.readthedocs.io
.. _LAMMPS: https://docs.lammps.org
.. |ReadTheDocs| image:: https://readthedocs.org/projects/relentless/badge/?version=latest
   :target: https://relentless.readthedocs.io/en/latest/?badge=latest
