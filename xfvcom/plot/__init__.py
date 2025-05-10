"""
xfvcom.plot sub-package public API.
"""

from __future__ import annotations

# ------------------------------------------------------------------
# Re-export core plotting classes / helpers
# ------------------------------------------------------------------
from .config import FvcomPlotConfig
from .core import FvcomPlotter  # ← 追加すると便利
from .markers import make_node_marker_post

# ------------------------------------------------------------------
# Public symbol table
# ------------------------------------------------------------------
__all__: list[str] = [
    "FvcomPlotConfig",
    "FvcomPlotter",
    "make_node_marker_post",
]
