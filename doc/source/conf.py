# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import datetime
import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------

project = "relentless"
year = datetime.date.today().year
copyright = f"2021-{year}, Auburn University"
author = "Michael P. Howard"
version = "0.1.0"
release = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "IPython.sphinxext.ipython_console_highlighting",
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_design",
]

templates_path = ["_templates"]

exclude_patterns = []

default_role = "any"

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["theme_overrides.css"]
html_theme_options = {
    "favicons": [
        {
            "rel": "icon",
            "href": "relentless_icon.svg",
        }
    ],
    "logo": {
        "image_light": "relentless_logo.svg",
        "image_dark": "relentless_logo_dark.svg",
        "alt_text": "relentless",
    },
}

# -- Options for autodoc & autosummary ---------------------------------------

autosummary_generate = True

autodoc_default_options = {"inherited-members": None, "special-members": False}

# -- Options for intersphinx -------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "networkx": ("https://networkx.org/documentation/stable", None),
    "mpi4py": ("https://mpi4py.readthedocs.io/en/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "hoomd": ("https://hoomd-blue.readthedocs.io/en/v2.9.7", None),
    "lammps": ("https://docs.lammps.org", None),
    "freud": ("https://freud.readthedocs.io/en/stable", None),
}
