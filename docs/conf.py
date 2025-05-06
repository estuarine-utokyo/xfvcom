# docs/conf.py
"""
Sphinx configuration for xfvcom.
Only the minimal settings required for HTML output.
"""
import pathlib
import sys

# Add project root to sys.path so autodoc can import xfvcom
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

project = "xfvcom"
author = "Jun Sasaki"
release = "0.2.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # Google / NumPy docstring support
    "sphinx_autodoc_typehints",  # Render type hints
]

html_theme = "furo"  # Clean, responsive theme
html_title = f"{project} {release} documentation"
autoclass_content = "both"  # Show __init__ docstrings
