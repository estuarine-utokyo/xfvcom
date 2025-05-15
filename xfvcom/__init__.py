# ------------------------------------------------------------------
# 0) lightweight first: pure-utility modules (no heavy imports)
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# 1) core analysis / I/O
# ------------------------------------------------------------------
from .analysis import FvcomAnalyzer
from .decorators import precedence
from .grid import FvcomGrid, get_grid, read_grid
from .io import FvcomDataLoader, parse_river_namelist  # noqa: F401

# ------------------------------------------------------------------
# 2) plotting sub-package (ordered: config → core → utils)
# ------------------------------------------------------------------
from .plot.config import FvcomPlotConfig
from .plot.core import FvcomPlotter
from .plot.markers import make_node_marker_post  # ← 直接 import
from .plot.utils import create_anim_2d_plot
from .plot_options import FvcomPlotOptions
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

# ------------------------------------------------------------------
# 3) public symbol table
# ------------------------------------------------------------------
__all__: list[str] = [
    # plotting
    "FvcomPlotter",
    "FvcomPlotConfig",
    "FvcomPlotOptions",
    "create_anim_2d_plot",
    "make_node_marker_post",
    # analysis / I/O
    "FvcomDataLoader",
    "FvcomAnalyzer",
    "parse_river_namelist",
    # helpers
    "FrameGenerator",
    "PlotHelperMixin",
    "create_gif",
    "create_gif_from_frames",
    "apply_xlim_ylim",
    "evaluate_model_scores",
    "generate_test_data",
    # misc
    "precedence",
    # grid
    "FvcomGrid",
    "get_grid",
    "read_grid",
]

__version__ = "0.2.0"
