# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------

project = 'relentless'
copyright = '2021, Auburn University'
author = 'Michael P. Howard'
version = '0.1.0'
release = '0.1.0'


# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo'
]

templates_path = ['_templates']

exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_context = {
    'css_files': [
        '_static/theme_overrides.css',
        ],
    }

# -- Options for autodoc & autosummary ---------------------------------------

autodoc_default_options = {
    'inherited-members': False
}

autosummary_generate = False

# -- Options for intersphinx -------------------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'networkx': ('https://networkx.org/documentation/stable', None),
    'mpi4py': ('https://mpi4py.readthedocs.io/en/stable', None),
    'numpy': ('https://numpy.org/doc/stable', None)
}
