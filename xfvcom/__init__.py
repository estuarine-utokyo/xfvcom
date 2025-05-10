"""
xfvcom top-level package
Re-export public APIs after file split.
"""

from __future__ import annotations

from .analysis import FvcomAnalyzer
from .decorators import precedence
from .io import FvcomDataLoader, parse_river_namelist  # noqa: F401

# ------------------------------------------------------------------
# 1) new canonical location  (xfvcom.plot.core)
# ------------------------------------------------------------------
from .plot.config import FvcomPlotConfig
from .plot.core import FvcomPlotter
from .plot.utils import create_anim_2d_plot
from .plot_options import FvcomPlotOptions

# ------------------------------------------------------------------
# 2) helpers / utilities (unchanged)
# ------------------------------------------------------------------
from .utils.helpers import (
    FrameGenerator,
    PlotHelperMixin,
    create_gif,
    create_gif_from_frames,
)
from .utils.helpers_utils import (
    apply_xlim_ylim,
    evaluate_model_scores,
    generate_test_data,
)

__all__ = [
    "FvcomPlotter",
    "FvcomDataLoader",
    "FvcomAnalyzer",
    "FvcomPlotConfig",
    "PlotHelperMixin",
    "FrameGenerator",
    "create_gif",
    "create_gif_from_frames",
    "apply_xlim_ylim",
    "evaluate_model_scores",
    "generate_test_data",
    "create_anim_2d_plot",
    "FvcomPlotOptions",
    "precedence",
    "parse_river_namelist",
]
