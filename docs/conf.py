# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'scDoRI'
copyright = '2025, Manu Saraswat'
author = 'Manu Saraswat'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode','nbsphinx',"sphinx_design"
    ]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = "furo"
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "sidebar_hide_name": True,
    "github_url": "https://github.com/saraswatmanu/scDoRI",
 "navbar_end": ["navbar-icon-links"],

    }

html_static_path = ['_static']
html_context = {
    "display_github": True,  # Integrate GitHub
    "github_user": "saraswatmanu",
    "github_repo": "scDoRI",
    "github_version": "main",
    "conf_py_path": "/docs/",  
}
# -- Path setup --------------------------------------------------------------

import os
import sys
sys.path.insert(0, os.path.abspath('..'))
