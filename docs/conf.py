# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].
"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""
# Standard Library
import os
import pathlib
import re
import sys
from typing import Dict, List, Optional

# Third Party
from sphinx.ext import apidoc

# -- Project information -----------------------------------------------------

project = "industreallib"
copyright = "2022, NVIDIA"
author = "NVIDIA Seattle Robotics Lab"
generate_api_docs = True

# -- Auto-generated API documentation ----------------------------------------

if generate_api_docs:
    # If extensions (or modules to document with autodoc) are in another directory,
    # add these directories to sys.path here. If the directory is relative to the
    # documentation root, use os.path.abspath to make it absolute, like shown here.

    root = pathlib.Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "src/nvidia"))

    # -- Run sphinx-apidoc -------------------------------------------------------
    # This hack is necessary since RTD does not issue `sphinx-apidoc` before running
    # `sphinx-build -b html docs _build/docs`.
    # See Issue: https://github.com/[[rtfd/readthedocs.org/issues/1139

    output_dir = os.path.join(root, "docs", "_api")
    module_dir = os.path.join(root, "src", "nvidia")

    apidoc_args = [
        "--implicit-namespaces",
        "--force",
        "--separate",
        "--module-first",
        "--no-toc",
        "-o",
        f"{output_dir}",
        f"{module_dir}",
    ]

    try:
        apidoc.main(apidoc_args)
        print("Running `sphinx-apidoc` complete!")
    except Exception as e:
        print(f"ERROR: Running `sphinx-apidoc` failed!\n{e}")

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions that are shipped with Sphinx (named 'sphinx.ext.*') or your
# custom ones.

# TODO (roflaherty): Add 'sphinx.ext.napoleon'
# See: https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
# NOTE: For more extensions, see
# * https://www.sphinx-doc.org/en/master/usage/extensions/index.html
# * https://matplotlib.org/sampledoc/extensions.html
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.graphviz",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx_rtd_theme",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns: List[str] = []

# List of warning types to suppress
suppress_warnings: List[str] = []

# The reST default role (used for this markup: `text`) to use for all
# documents.
default_role: Optional[str] = "code"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
# html_theme = 'bizstyle'
# html_theme = 'agogo'
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_css_files = ["custom.css"]

html_logo = "_static/nvidia_logo.png"

# -- Options for extensions --------------------------------------------------

# sphinx.ext.autodoc options
# --------------------------
autoclass_content = "both"
autodoc_class_signature = "mixed"
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_warningiserror = False
suppress_warnings.extend(["autodoc"])

# mathjax options
# ---------------
# NOTE (roflaherty): See
# https://www.sphinx-doc.org/en/master/usage/extensions/math.html#confval-mathjax_config
# http://docs.mathjax.org/en/latest/options/index.html#configuring-mathjax
# https://stackoverflow.com/a/60497853
mathjax3_config: Dict[str, dict] = {"tex": {"macros": {}}}

with open("mathsymbols.tex", "r") as f:
    for line in f:
        macros = re.findall(r"\\(DeclareRobustCommand|newcommand){\\(.*?)}(\[(\d)\])?{(.+)}", line)
        for macro in macros:
            if len(macro[2]) == 0:
                mathjax3_config["tex"]["macros"][macro[1]] = "{" + macro[4] + "}"
            else:
                mathjax3_config["tex"]["macros"][macro[1]] = ["{" + macro[4] + "}", int(macro[3])]


# sphinx.ext.todo options
# -----------------------
todo_include_todos = True

# sphinx_rtd_theme options
# ------------------------
html_theme_options = {"navigation_depth": 1}
