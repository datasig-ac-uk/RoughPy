# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'RoughPy'
copyright = '2023, The RoughPy Authors'
author = 'The RoughPy Authors'
release = '0.0.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx_design",
    "sphinx.ext.todo",
    "nbsphinx",
    'myst_parser',
    'IPython.sphinxext.ipython_console_highlighting',
    'sphinxcontrib.bibtex',
    "sphinxcontrib.video",
]

bibtex_bibfiles = ['references.bib']

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'restructuredtext',
    '.md': 'markdown'
}

templates_path = ['_templates']
exclude_patterns = ['_build', '**.ipynb_checkpoints']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']


intersphinx_mapping = {'python': ('https://docs.python.org/3', None)}

# Display todos by setting to True
todo_include_todos = True
