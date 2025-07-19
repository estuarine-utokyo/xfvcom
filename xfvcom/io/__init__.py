# xfvcom/io/__init__.py
"""
I/O sub-package of xfvcom.

* core.py      – former standalone io.py
* river_nml.py – river namelist parser
"""

from __future__ import annotations

# ---------------------------------------------------------------------
# Public symbol table – declare *before* the imports so mypy is happy
# ---------------------------------------------------------------------
__all__: list[str] = []  # explicit annotation for mypy

# ---------------------------------------------------------------------
# Re-export everything from the former `io.py`
# ---------------------------------------------------------------------
from .core import *  # noqa: F401,F403

# core.py may itself define __all__; if so, merge it.
_core_all: list[str] | None = globals().get("__all__")  # after star-import
if _core_all:  # truthy & already a list[str]
    __all__.extend(_core_all)

from .groundwater_nc_generator import GroundwaterNetCDFGenerator  # noqa: F401
from .input_loader import FvcomInputLoader  # noqa: F401

# ---------------------------------------------------------------------
# New helper(s)
# ---------------------------------------------------------------------
from .river_nml import parse_river_namelist  # noqa: F401

__all__.append("parse_river_namelist")
__all__.append("GroundwaterNetCDFGenerator")
__all__.append("FvcomInputLoader")
