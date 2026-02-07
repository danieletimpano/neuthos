# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# docs/source/conf.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]      # project root
sys.path.insert(0, str(ROOT / "src"))           # so `import neuthos` works

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'NEUTHOS'
copyright = '2026, Daniele Timpano'
author = 'Daniele Timpano'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",          
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "../../examples/**"]

autosummary_generate = True

# Autodoc options
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# If docs build fails because deps are missing, keep these.
# Remove ones you *do* have installed to get better signatures.
autodoc_mock_imports = [
    "numpy",
    "pandas",
    "matplotlib",
    "scipy",
]

# -- Options for HTML output -------------------------------------------------
html_logo = "_static/neuthos_logo.png"
html_theme = "pydata_sphinx_theme"
html_css_files = ["custom.css"]

html_theme_options = {
    "navbar_align": "content",
    "show_nav_level": 1,
}