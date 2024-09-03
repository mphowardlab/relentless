# relentless

[![PyPI version](https://img.shields.io/pypi/v/relentless)](https://pypi.org/project/relentless)
[![PyPI downloads](https://img.shields.io/pypi/dm/relentless)](https://pypi.org/project/relentless)
[![Conda downloads](https://img.shields.io/conda/dn/conda-forge/relentless)](https://anaconda.org/conda-forge/relentless)
[![Read the Docs](https://readthedocs.org/projects/relentless/badge/?version=stable)](https://relentless.readthedocs.io/en/stable/?badge=stable)


## Overview

`relentless` is a Python package for executing molecular simulations in larger
computational workflows. The simulation protocol is specified as a reproducible,
human-readable recipe that is run natively in popular engines such as
[LAMMPS](https://docs.lammps.org) and
[HOOMD-blue](https://hoomd-blue.readthedocs.io). `relentless` also has robust
features for optimization with simulations, including independent and dependent
variables with automatic differentiation, objective functions of simulations,
and optimization methods. All features are readily extensible through a clean
hierarchy of abstract objects, enabling you to quickly use your own simulation
code or optimization objective. Our initial goal was to enable optimization of
the `rel`ative `ent`ropy of structural coarse-graining and materials design:
with `less` code.


## Resources

- [Documentation](https://relentless.readthedocs.io):
  Installation, examples, and commands.
- [Source code](https://github.com/mphowardlab/relentless):
  Download or contribute on GitHub.
- [Issue tracker](https://github.com/mphowardlab/relentless/issues):
  Report issues or request features.


## Installation

Install `relentless` from PyPI using `pip`:

    pip install relentless

or from conda-forge using `conda`:

    conda install -c conda-forge relentless


## Contributing

Contributions are welcomed and appreciated! Fork and create a pull request on
[GitHub](https://github.com/mphowardlab/relentless). We value the input and
experiences all users and contributors bring to `relentless`.
