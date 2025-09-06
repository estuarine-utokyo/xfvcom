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

# Plotly utilities (optional, only if plotly is installed)
try:
    from .plotly_utils import (
        plot_timeseries_comparison,
        plot_timeseries_multi_variable,
        create_river_extension_plot,
        print_plotly_instructions,
    )
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# ------------------------------------------------------------------
# Public symbol table
# ------------------------------------------------------------------
__all__: list[str] = [
    "FvcomPlotConfig",
    "FvcomPlotter",
    "make_node_marker_post",
]

# Add plotly functions to __all__ if available
if PLOTLY_AVAILABLE:
    __all__.extend([
        "plot_timeseries_comparison",
        "plot_timeseries_multi_variable", 
        "create_river_extension_plot",
        "print_plotly_instructions",
    ])
