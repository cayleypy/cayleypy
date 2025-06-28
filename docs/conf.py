# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

from sphinx_markdown_parser.parser import MarkdownParser

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def setup(app):
    app.add_source_parser(MarkdownParser)
    app.add_config_value("pandoc_use_parser", "markdown", True)
    app.connect("autodoc-process-signature", autodoc_process_signature)
    app.add_css_file("custom.css")


# -- Project information -----------------------------------------------------

project = "InferLO"
copyright = "2020, InferLO developers"
author = "InferLO developers"

# The full version, including alpha/beta/rc tags
release = "0.1.0"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.linkcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",  # Optional: for summary tables
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    "generated/inferlo.rst",  # No need in page for root __init__.py.
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

# Allow markdown includes.
# http://www.sphinx-doc.org/en/master/markdown.html
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Home page for documentation.
master_doc = "index"

# -- Extension configuration -------------------------------------------------

# Generate subpages for reference docs automatically.
# http://www.sphinx-doc.org/en/master/ext/autosummary.html#generating-stub-pages-automatically
autosummary_generate = True

# Autodoc configuration.
# See https://www.sphinx-doc.org/en/2.0/usage/extensions/autodoc.html

# Autodoc options.
autodoc_default_options = {
    "members": None,  # Enables docs for class members.
}
autodoc_typehints = "signature"


def autodoc_process_signature(app, what, name, obj, options, signature, return_annotation):
    if what == "class":
        return None, None


import inspect
import cayleypy


def linkcode_resolve(domain, info):
    if domain != "py" or not info["module"]:
        return None
    try:
        mod = sys.modules.get(info["module"])
        obj = mod
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)
        fn = inspect.getsourcefile(obj)
        source, lineno = inspect.getsourcelines(obj)
    except Exception:
        return None

    fn = os.path.relpath(fn, start=os.path.dirname(cayleypy.__file__))

    return f"https://github.com/cayleypy/cayleypy/blob/main/{fn}#L{lineno}-L{lineno + len(source) - 1}"
