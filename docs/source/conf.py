
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Bayesian Linear Regression'
copyright = '2024, Joaquin Rapela'
author = 'Joaquin Rapela'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx_gallery.gen_gallery',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = []

sphinx_gallery_conf = {
    'examples_dirs': '../../examples/sphinx-gallery/',   # path to your example scripts
    'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
    'binder': {
        # Required keys
        'org': 'joacorapela',
        'repo': 'bayesianLinearRegression',
        'branch': 'gh-pages', # Can be any branch, tag, or commit hash. Use a branch that hosts your docs.
        'binderhub_url': 'https://mybinder.org', # Any URL of a binderhub deployment. Must be full URL (e.g. https://mybinder.org).
        'dependencies': '../requirements.txt',
        # Optional keys
        # 'filepath_prefix': '<prefix>' # A prefix to prepend to any filepaths in Binder links.
        # 'notebooks_dir': '<notebooks-directory-name>' # Jupyter notebooks for Binder will be copied to this directory (relative to built documentation root).
        # 'use_jupyter_lab': <bool> # Whether Binder links should start Jupyter Lab instead of the Jupyter Notebook interface.
    }
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
