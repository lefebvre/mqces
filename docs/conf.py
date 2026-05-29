"""Sphinx configuration for the mqces documentation site."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# -- Path setup --------------------------------------------------------------
# Allow autodoc to import the Python package without a full editable install
# (CI installs the wheel; local builds may not). The wheel-install path takes
# precedence when present.
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "python"))

# -- Project information -----------------------------------------------------
project = "mqces"
author = "Jordan P. Lefebvre"
copyright = "2026, Jordan P. Lefebvre"

# Version is read from the CMake project() line so the docs never drift from
# the library version. Fallback to "dev" if the file is unreadable.
def _project_version() -> str:
    cmake = _repo_root / "CMakeLists.txt"
    try:
        for line in cmake.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("VERSION "):
                return stripped.split()[1]
    except OSError:
        pass
    return "dev"

version = _project_version()
release = version

# -- General configuration ---------------------------------------------------
extensions = [
    "breathe",
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
master_doc = "index"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- MyST (Markdown) ---------------------------------------------------------
myst_enable_extensions = [
    "amsmath",        # $$..$$ blocks for paper equations
    "dollarmath",     # $..$ inline math
    "deflist",
    "colon_fence",    # ::: blocks for admonitions
    "attrs_inline",
]
myst_heading_anchors = 3

# -- Breathe (Doxygen XML bridge) --------------------------------------------
# `breathe_projects` is populated dynamically: GitHub Actions and CMake both
# set MQCES_DOXYGEN_XML_DIR to the directory containing index.xml. Local
# builds without that variable default to docs/_build/doxygen/xml.
_default_xml = _repo_root / "docs" / "_build" / "doxygen" / "xml"
breathe_projects = {
    "mqces": os.environ.get("MQCES_DOXYGEN_XML_DIR", str(_default_xml)),
}
breathe_default_project = "mqces"
breathe_default_members = ("members", "undoc-members")
breathe_show_define_initializer = True

# -- Napoleon (NumPy-style Python docstrings) --------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- Autodoc -----------------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"

# Autodoc imports `mqces` to introspect it. The nanobind core extension
# (`_mqces_core`) may be absent during a docs-only build, so mock it.
autodoc_mock_imports = ["mqces._mqces_core"]

# -- Intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# -- HTML output -------------------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
html_title = f"mqces {version}"
html_theme_options = {
    "source_repository": "https://github.com/lefebvre/mqces/",
    "source_branch": "master",
    "source_directory": "docs/",
    "navigation_with_keys": True,
}

# -- Misc --------------------------------------------------------------------
# Warn on missing references so PR builds surface broken cross-links early.
nitpicky = False  # too aggressive against Eigen/STL types; keep off for now

