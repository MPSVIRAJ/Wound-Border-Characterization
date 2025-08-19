# Configuration file for the Sphinx documentation builder.
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Wound Border Characterization'
copyright = '2025, Sameera Viraj'
author = 'Sameera Viraj'
release = '1.0.0'
html_short_title = "Wound Characterization"
# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
]

autodoc_member_order = "bysource"

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'furo'
html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_theme_options = {
    "logo": {
        "text": "Wound Border Characterization",
    },

    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/MPSVIRAJ/Wound-Border-Caracterization", # Your GitHub URL
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
    ],
}

