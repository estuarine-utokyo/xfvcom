"""
xfvcom.plot sub-package public API.
"""

from __future__ import annotations

from ._timeseries_utils import (
    get_member_color,
    get_member_colors,
)

# ------------------------------------------------------------------
# Re-export core plotting classes / helpers
# ------------------------------------------------------------------
from .config import FvcomPlotConfig
from .core import FvcomPlotter  # ← 追加すると便利
from .dye_timeseries import plot_dye_timeseries_stacked
from .markers import make_node_marker_post
from .timeseries import (
    apply_smart_time_ticks,
    plot_ensemble_statistics,
    plot_ensemble_timeseries,
)

# Plotly utilities (optional, only if plotly is installed)
try:
    from .plotly_utils import (
        create_river_extension_plot,
        plot_timeseries_comparison,
        plot_timeseries_multi_variable,
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
    "apply_smart_time_ticks",
    "plot_ensemble_timeseries",
    "plot_ensemble_statistics",
    "plot_dye_timeseries_stacked",
    "get_member_color",
    "get_member_colors",
]

# Add plotly functions to __all__ if available
if PLOTLY_AVAILABLE:
    __all__.extend(
        [
            "plot_timeseries_comparison",
            "plot_timeseries_multi_variable",
            "create_river_extension_plot",
            "print_plotly_instructions",
        ]
    )
