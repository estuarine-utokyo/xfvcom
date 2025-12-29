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
from .source_contribution import (
    create_source_contribution_plot,
    load_dye_timeseries_multi_source,
    plot_source_contribution_stack,
)
from .source_map import (
    SOURCE_NODES,
    SOURCE_TYPES,
    get_source_coordinates,
    load_grid_coordinates,
    plot_source_map,
)
from .timeseries import (
    apply_smart_time_ticks,
    plot_ensemble_statistics,
    plot_ensemble_timeseries,
    plot_timeseries,
    plot_timeseries_multi,
)
from .variable_meta import (
    VARIABLE_REGISTRY,
    VariableMeta,
    get_label,
    get_variable_meta,
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
    "plot_timeseries",
    "plot_timeseries_multi",
    "plot_ensemble_timeseries",
    "plot_ensemble_statistics",
    "plot_dye_timeseries_stacked",
    "get_member_color",
    "get_member_colors",
    # Source contribution plots
    "create_source_contribution_plot",
    "load_dye_timeseries_multi_source",
    "plot_source_contribution_stack",
    # Source map plots
    "SOURCE_NODES",
    "SOURCE_TYPES",
    "get_source_coordinates",
    "load_grid_coordinates",
    "plot_source_map",
    # Variable metadata
    "VariableMeta",
    "get_variable_meta",
    "get_label",
    "VARIABLE_REGISTRY",
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

# hvPlot utilities (optional, only if hvplot is installed)
try:
    from .hvplot_utils import (
        InteractivePlotter,
        hvplot_timeseries,
    )

    HVPLOT_AVAILABLE = True
except ImportError:
    HVPLOT_AVAILABLE = False

# Add hvplot functions to __all__ if available
if HVPLOT_AVAILABLE:
    __all__.extend(
        [
            "InteractivePlotter",
            "hvplot_timeseries",
        ]
    )
